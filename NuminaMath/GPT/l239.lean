import Mathlib

namespace total_serving_time_l239_239204

def patients : ℕ := 12
def special_diet_fraction : ℚ := 1/3
def standard_care_time : ℕ := 5
def time_increase_fraction : ℚ := 1/5 -- equivalent to 20%

theorem total_serving_time :
  let special_diet_patients := (special_diet_fraction * patients).toNat,
      std_patients := patients - special_diet_patients,
      additional_time := (time_increase_fraction * standard_care_time).toNat,
      time_special := standard_care_time + additional_time,
      total_time_std := std_patients * standard_care_time,
      total_time_special := special_diet_patients * time_special in
  total_time_std + total_time_special = 64 :=
by
  sorry

end total_serving_time_l239_239204


namespace sum_of_squares_of_solutions_l239_239869

theorem sum_of_squares_of_solutions : 
  let a := 1 / 2016 in
  let expr := (x : ℝ) → x^2 - x + a in
  set s := { x : ℝ | abs (expr x) = a } in
  ∑ x in s, x^2 = 1007 / 504 :=
sorry

end sum_of_squares_of_solutions_l239_239869


namespace system_of_equations_solution_l239_239023

theorem system_of_equations_solution {a x y : ℝ} (h1 : y > 0) (h2 : x ≥ 0) :
  (∃ x y, x - 4 = a * (y^3 - 2) ∧ (2 * x) / (|y^3| + y^3) = sqrt x) ↔ 
  (a ∈ Set.Ioo (-∞) 0 ∪ Set.Ioo 2 ∞) :=
sorry

end system_of_equations_solution_l239_239023


namespace probability_of_triangle_or_circle_l239_239653

/-- The total number of figures -/
def total_figures : ℕ := 10

/-- The number of triangles -/
def triangles : ℕ := 3

/-- The number of circles -/
def circles : ℕ := 3

/-- The number of figures that are either triangles or circles -/
def favorable_figures : ℕ := triangles + circles

/-- The probability that the chosen figure is either a triangle or a circle -/
theorem probability_of_triangle_or_circle : (favorable_figures : ℚ) / (total_figures : ℚ) = 3 / 5 := 
by
  sorry

end probability_of_triangle_or_circle_l239_239653


namespace previous_day_visitors_l239_239810

-- Define the number of visitors on the day Rachel visited
def visitors_on_day_rachel_visited : ℕ := 317

-- Define the difference in the number of visitors between the day Rachel visited and the previous day
def extra_visitors : ℕ := 22

-- Prove that the number of visitors on the previous day is 295
theorem previous_day_visitors : visitors_on_day_rachel_visited - extra_visitors = 295 :=
by
  sorry

end previous_day_visitors_l239_239810


namespace largest_lcm_l239_239273

theorem largest_lcm :
  let l4 := Nat.lcm 18 3
  let l5 := Nat.lcm 18 6
  let l6 := Nat.lcm 18 9
  let l7 := Nat.lcm 18 12
  let l8 := Nat.lcm 18 15
  let l9 := Nat.lcm 18 18
  in max (max (max (max (max l4 l5) l6) l7) l8) l9 = 90 := by
    sorry

end largest_lcm_l239_239273


namespace danielles_rooms_l239_239572

variable (rooms_heidi rooms_danielle : ℕ)

theorem danielles_rooms 
  (h1 : rooms_heidi = 3 * rooms_danielle)
  (h2 : 2 = 1 / 9 * rooms_heidi) :
  rooms_danielle = 6 := by
  -- Proof omitted
  sorry

end danielles_rooms_l239_239572


namespace second_largest_angle_l239_239166

theorem second_largest_angle (A B C U I : Type) [triangle ABC] [circumcenter U ABC] [incenter I ABC]
  (hABC_scalene : scalene ABC)
  (h_intersection : ∃ D, D ≠ C ∧ (angle_bisector γ ∩ circumcircle ABC).second = D ∧ perpendicular_bisector UI) 
  (angle_bisector γ := ∠ACB_bisector)
  (γ := angle ACB (triangle.angles ABC)) :
  γ = 60 → is_second_largest_angle γ ABC :=
sorry

end second_largest_angle_l239_239166


namespace mango_rate_l239_239934

theorem mango_rate 
  (grapes_kg : ℕ) (grapes_rate : ℕ) 
  (mangoes_kg : ℕ) (total_paid : ℕ) 
  (grapes_kg = 8) 
  (grapes_rate = 80) 
  (mangoes_kg = 9) 
  (total_paid = 1135) :
  (mangoes_rate : ℕ) (495 = total_paid - (grapes_kg * grapes_rate)) :
  mangoes_rate = 55 /- where 495 = total_paid - grapes_kg * grapes_rate -/
  :=
sorry

end mango_rate_l239_239934


namespace imaginary_part_of_complex_number_l239_239079

theorem imaginary_part_of_complex_number (i : ℂ) (h : i = Complex.I) : Complex.imag (1 - 2 * i) = -2 := by
  sorry

end imaginary_part_of_complex_number_l239_239079


namespace hyperbola_focal_length_l239_239496

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239496


namespace part1_inequality_part2_range_of_a_l239_239530

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239530


namespace wizard_possible_configuration_l239_239155

theorem wizard_possible_configuration (N : ℕ) :
  (∃ f : Fin N → Fin N,
    ∀ i j : Fin N, i.val < j.val → ¬ (f j = i)) ∧
  (∃ (i : Fin N), ∀ j: Fin N, ∃ k: List (Fin N), k.head = i ∧ k.last = j ∧ (∀ l: ℕ, l < List.length k - 1 -> (k.nth l).val < (k.nth (l + 1)).val)) ∧
  (∃ (i : Fin N), ∀ k: List (Fin N), k.head = i ∧ ∃ j: Fin N, j ∉ k.tail) ∧
  (∃ k: List (Fin N), k.length = N  ∧  ∀ l : ℕ, l < List.length k - 1 -> (k.nth l).val < (k.nth (l + 1)).val) ∧
  N! = List.perm_count N := by
  sorry

end wizard_possible_configuration_l239_239155


namespace problem1_problem2_l239_239522

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239522


namespace propositions_correct_l239_239506

-- Defining the propositions
def proposition1 (f : ℝ → ℝ) (b c : ℝ) : Prop :=
  (∀ x, f(x) = x * abs x + b * x + c) ∧ (∀ x, f (-x) = -f x → c = 0)

def proposition2 : Prop :=
  (∀ x > 0, (2 : ℝ)^(-x) = x → x = 2^(-y) → y = -log 2 x)

def proposition3 (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∃ x, f x = log10 (x^2 + a * x - a)) ∧ (range f = set.univ → a ≤ -4 ∨ a ≥ 0)

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f(x - 1) = f(-(x - 1))) ∧ (∀ x, (f x) symmetric about axis 1)

-- The proof problem
theorem propositions_correct :
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4) :=
by
  -- Skip the proof
  sorry

end propositions_correct_l239_239506


namespace volume_ratio_of_cubes_l239_239277

-- Given conditions
def edge_length_smaller_cube : ℝ := 6
def edge_length_larger_cube : ℝ := 12

-- Problem statement
theorem volume_ratio_of_cubes : 
  (edge_length_smaller_cube / edge_length_larger_cube) ^ 3 = (1 / 8) := 
by
  sorry

end volume_ratio_of_cubes_l239_239277


namespace trapezoid_height_proof_l239_239602

-- Given lengths of the diagonals and the midline of the trapezoid
def diagonal1Length : ℝ := 6
def diagonal2Length : ℝ := 8
def midlineLength : ℝ := 5

-- Target to prove: Height of the trapezoid
def trapezoidHeight : ℝ := 4.8

theorem trapezoid_height_proof :
  ∀ (d1 d2 m : ℝ), d1 = diagonal1Length → d2 = diagonal2Length → m = midlineLength → trapezoidHeight = 4.8 :=
by intros d1 d2 m hd1 hd2 hm; sorry

end trapezoid_height_proof_l239_239602


namespace cheenu_time_difference_l239_239754

theorem cheenu_time_difference :
  let boy_distance : ℝ := 18
  let boy_time_hours : ℝ := 4
  let old_man_distance : ℝ := 12
  let old_man_time_hours : ℝ := 5
  let hour_to_minute : ℝ := 60
  
  let boy_time_minutes := boy_time_hours * hour_to_minute
  let old_man_time_minutes := old_man_time_hours * hour_to_minute

  let boy_time_per_mile := boy_time_minutes / boy_distance
  let old_man_time_per_mile := old_man_time_minutes / old_man_distance
  
  old_man_time_per_mile - boy_time_per_mile = 12 :=
by sorry

end cheenu_time_difference_l239_239754


namespace initial_cheesecakes_on_display_l239_239778

theorem initial_cheesecakes_on_display (D : ℕ) (in_fridge : ℕ) (sold_from_display : ℕ) (total_left : ℕ) :
  in_fridge = 15 →
  sold_from_display = 7 →
  total_left = 18 →
  D = (total_left - in_fridge) + sold_from_display →
  D = 10 :=
begin
  intros in_fridge_cond sold_from_display_cond total_left_cond D_def,
  sorry
end

end initial_cheesecakes_on_display_l239_239778


namespace elisa_lap_time_improvement_l239_239016

-- Define initial and current conditions
def initial_lap_time : ℝ := 25 / 10
def current_lap_time : ℝ := 24 / 12

-- Prove the improvement is 1/2 minute
theorem elisa_lap_time_improvement : initial_lap_time - current_lap_time = 1 / 2 := by
  sorry

end elisa_lap_time_improvement_l239_239016


namespace num_ways_seating_l239_239149

theorem num_ways_seating (n : ℕ) (h : n = 6) : (nat.factorial n) / n = nat.factorial (n - 1) :=
by 
  rw h
  calc
    (nat.factorial 6) / 6 = 720 / 6    : by norm_num
                      ... = 120        : by norm_num
                      ... = nat.factorial 5 : by norm_num

end num_ways_seating_l239_239149


namespace part1_solution_part2_solution_l239_239553

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239553


namespace negation_of_proposition_l239_239563

open Real

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > sin x) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_proposition_l239_239563


namespace hyperbola_focal_length_l239_239462

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239462


namespace find_sum_of_pqr_l239_239301

theorem find_sum_of_pqr (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) 
(h4 : q = p * (4 - p)) (h5 : r = q * (4 - q)) (h6 : p = r * (4 - r)) : 
p + q + r = 6 :=
sorry

end find_sum_of_pqr_l239_239301


namespace problem1_problem2_problem3_l239_239885

open Real

-- Definitions based on conditions
noncomputable
def f (x : ℝ) (m : ℝ) : ℝ := log x / log m

def isArithmeticSeq (f : ℕ → ℝ) (a d : ℝ) : Prop :=
  ∀ n : ℕ, f n = a + d * (n - 1)

def isGeometricSeq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Problem 1: Prove that {a_n} is a geometric series
theorem problem1 {m : ℝ} (h₁ : m > 0) (h₂ : m ≠ 1)
  (f_arith_seq : ∀ n : ℕ, f (a n) m = 4 + 2 * (n - 1)) :
  isGeometricSeq a :=
sorry

-- Problem 2: Find S_n when m = sqrt(2)
noncomputable
def b (n : ℕ) (a : ℕ → ℝ) (f : ℕ → ℝ) : ℝ := a n * f n

noncomputable
def S (n : ℕ) (b : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range n, b i

theorem problem2 {m : ℝ} (h₁ : m = √2) (a_n : ℕ → ℝ)
  (b_n : ℕ → ℝ) (h₂ : ∀ n, b_n n = a_n n * f n) :
  S n b_n = 2^(n + 3) * n :=
sorry

-- Problem 3: Range of m for strictly increasing c_n
noncomputable
def c (n : ℕ) (a : ℕ → ℝ) : ℝ := a n * log (a n)

theorem problem3 (a_n : ℕ → ℝ)
  (h : ∀ n, a_n (n + 1) * log (a_n (n + 1)) > a_n n * log (a_n n)) :
  0 < m ∧ m < (√6 / 3) ∨ m > 1 :=
sorry

end problem1_problem2_problem3_l239_239885


namespace find_b2013_minus_a2013_l239_239408

noncomputable theory

variables {a b : ℝ}

theorem find_b2013_minus_a2013
    (h1 : {1, a + b, a} = ({0, b/a, b} : set ℝ))
    (h2 : a ≠ 0)
    (h3 : a + b = 0)
    (h4 : b = 1) :
  b^2013 - a^2013 = 2 :=
by
  sorry

end find_b2013_minus_a2013_l239_239408


namespace f_f_zero_l239_239508

def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * real.pi^2 - 4 :=
by sorry

end f_f_zero_l239_239508


namespace middle_integer_of_sum_is_120_l239_239711

-- Define the condition that three consecutive integers sum to 360
def consecutive_integers_sum_to (n : ℤ) (sum : ℤ) : Prop :=
  (n - 1) + n + (n + 1) = sum

-- The statement to prove
theorem middle_integer_of_sum_is_120 (n : ℤ) :
  consecutive_integers_sum_to n 360 → n = 120 :=
by
  sorry

end middle_integer_of_sum_is_120_l239_239711


namespace degree_f_plus_g_l239_239222

def f (z : ℚ) := a_3 * z ^ 3 + a_2 * z ^ 2 + a_1 * z + a_0
def g (z : ℚ) := b_2 * z ^ 2 + b_1 * z + b_0

theorem degree_f_plus_g (h1 : a_3 ≠ 0) : degree (f + g) = 3 :=
by sorry

end degree_f_plus_g_l239_239222


namespace hyperbola_focal_length_l239_239466

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239466


namespace zeros_of_g_l239_239509

def f : ℝ → ℝ
| x where x ≤ 0 := 2^x - 1
| x := f (x - 1) + 1

def g (x: ℝ) : ℝ := f(x) - x

noncomputable def a_n (n : ℕ) : ℝ := n - 1

theorem zeros_of_g (n : ℕ) : g (a_n n) = 0 := by
  sorry

end zeros_of_g_l239_239509


namespace work_completion_time_l239_239323

/-
Conditions:
1. A man alone can do the work in 6 days.
2. A woman alone can do the work in 18 days.
3. A boy alone can do the work in 9 days.

Question:
How long will they take to complete the work together?

Correct Answer:
3 days
-/

theorem work_completion_time (M W B : ℕ) (hM : M = 6) (hW : W = 18) (hB : B = 9) : 1 / (1/M + 1/W + 1/B) = 3 := 
by
  sorry

end work_completion_time_l239_239323


namespace focal_length_of_hyperbola_l239_239483

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239483


namespace even_number_of_odd_degrees_l239_239328

-- Define the village as a set of vertices
def vertex_set : FinSet := {v : Fin 101}

-- Define the friendship graph as undirected edges between vertices
def friendship_graph : SimpleGraph vertex_set := {
  adj := λ (a b : vertex_set), (--- Insert conditions for a and b being friends ---)
  symm := λ (a b : vertex_set) (hab : adj a b), adj b a, -- Friendship is mutual
  loopless := λ (a : vertex_set), ¬adj a a -- No self-loops
}

-- Define the degree function
def degree (v : vertex_set) : Nat := (friendship_graph.degree v)

-- Statement to be proven:
theorem even_number_of_odd_degrees (V : vertex_set) (E : Nat) :
  2 * E = (V.toFin∑ λ v, degree v) →
  ∃ S : Finset V, (∀ v ∈ S, degree v % 2 = 1) ∧ S.card % 2 = 0 :=
sorry

end even_number_of_odd_degrees_l239_239328


namespace cookies_count_l239_239404

theorem cookies_count :
  ∀ (Tom Lucy Millie Mike Frank : ℕ), 
  (Tom = 16) →
  (Lucy = Nat.sqrt Tom) →
  (Millie = 2 * Lucy) →
  (Mike = 3 * Millie) →
  (Frank = Mike / 2 - 3) →
  Frank = 9 :=
by
  intros Tom Lucy Millie Mike Frank hTom hLucy hMillie hMike hFrank
  have h1 : Tom = 16 := hTom
  have h2 : Lucy = Nat.sqrt Tom := hLucy
  have h3 : Millie = 2 * Lucy := hMillie
  have h4 : Mike = 3 * Millie := hMike
  have h5 : Frank = Mike / 2 - 3 := hFrank
  sorry

end cookies_count_l239_239404


namespace sin_double_angle_in_fourth_quadrant_l239_239939

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239939


namespace Beth_and_Jan_total_money_l239_239983

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end Beth_and_Jan_total_money_l239_239983


namespace part1_part2_l239_239542

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239542


namespace part1_inequality_part2_range_of_a_l239_239529

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239529


namespace focal_length_of_hyperbola_l239_239486

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239486


namespace part1_solution_set_part2_range_a_l239_239561

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239561


namespace at_least_one_even_diff_l239_239182

theorem at_least_one_even_diff (a1 a2 a3 : ℤ) (b1 b2 b3 : ℤ) (hb : {b1, b2, b3} = {a1, a2, a3}) :
  (a1 - b1) % 2 = 0 ∨ (a2 - b2) % 2 = 0 ∨ (a3 - b3) % 2 = 0 :=
sorry

end at_least_one_even_diff_l239_239182


namespace part_i_part_ii_l239_239078

variable (a : ℝ)
def p := ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem part_i (ha : p a) : a ≤ 1 :=
sorry

theorem part_ii (ha : ¬ (p a ∧ q a)) : a ∈ set.Ioo (-2:ℝ) 1 ∪ set.Ioi (1:ℝ) :=
sorry

end part_i_part_ii_l239_239078


namespace maximum_value_sin_cos_l239_239246

theorem maximum_value_sin_cos : ∀ x : ℝ, (sin x + cos x) ≤ sqrt 2 ∧ ∃ x : ℝ, (sin x + cos x) = sqrt 2 :=
by {
    sorry
}

end maximum_value_sin_cos_l239_239246


namespace top_triangle_is_multiple_of_5_l239_239334

-- Definitions of the conditions given in the problem

def lower_left_triangle := 12
def lower_right_triangle := 3

-- Let a, b, c, d be the four remaining numbers in the bottom row
variables (a b c d : ℤ)

-- Conditions that the sums of triangles must be congruent to multiples of 5
def second_lowest_row : Prop :=
  (3 - a) % 5 = 0 ∧
  (-a - b) % 5 = 0 ∧
  (-b - c) % 5 = 0 ∧
  (-c - d) % 5 = 0 ∧
  (2 - d) % 5 = 0

def third_lowest_row : Prop :=
  (2 + 2*a + b) % 5 = 0 ∧
  (a + 2*b + c) % 5 = 0 ∧
  (b + 2*c + d) % 5 = 0 ∧
  (3 + c + 2*d) % 5 = 0

def fourth_lowest_row : Prop :=
  (3 + 2*a + 2*b - c) % 5 = 0 ∧
  (-a + 2*b + 2*c - d) % 5 = 0 ∧
  (2 - b + 2*c + 2*d) % 5 = 0

def second_highest_row : Prop :=
  (2 - a + b - c + d) % 5 = 0 ∧
  (3 + a - b + c - d) % 5 = 0

def top_triangle : Prop :=
  (2 - a + b - c + d + 3 + a - b + c - d) % 5 = 0

theorem top_triangle_is_multiple_of_5 (a b c d : ℤ) :
  second_lowest_row a b c d →
  third_lowest_row a b c d →
  fourth_lowest_row a b c d →
  second_highest_row a b c d →
  top_triangle a b c d →
  ∃ k : ℤ, (2 - a + b - c + d + 3 + a - b + c - d) = 5 * k :=
by sorry

end top_triangle_is_multiple_of_5_l239_239334


namespace sin_double_angle_neg_l239_239945

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239945


namespace relationship_among_abc_l239_239097

variables (a b c : ℝ)
def a_def : a = 0.8^0.8 := by sorry
def b_def : b = 0.8^0.9 := by sorry
def c_def : c = 1.2^0.8 := by sorry

theorem relationship_among_abc (a b c : ℝ) (ha : a = 0.8^0.8) (hb : b = 0.8^0.9) (hc : c = 1.2^0.8) :
  c > a ∧ a > b :=
by sorry

end relationship_among_abc_l239_239097


namespace original_average_rent_is_800_l239_239231

def original_rent (A : ℝ) : Prop :=
  let friends : ℝ := 4
  let old_rent : ℝ := 800
  let increased_rent : ℝ := old_rent * 1.25
  let new_total_rent : ℝ := (850 * friends)
  old_rent * 4 - 800 + increased_rent = new_total_rent

theorem original_average_rent_is_800 (A : ℝ) : original_rent A → A = 800 :=
by 
  sorry

end original_average_rent_is_800_l239_239231


namespace hyperbola_focal_length_is_4_l239_239443

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239443


namespace part1_tangent_line_at_x2_part2_inequality_l239_239514

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

theorem part1_tangent_line_at_x2 (a : ℝ) (h_a : a = 2) :
  ∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = Real.exp 2 - 2 ∧ b = -(2 * Real.exp 2 - 7) := by
  sorry

theorem part2_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x a ≥ (7 / 4) * x^2) → a ≤ Real.exp 2 - 7 := by
  sorry

end part1_tangent_line_at_x2_part2_inequality_l239_239514


namespace find_k_l239_239932

open Real

noncomputable def a := (1, 1) : ℝ × ℝ
noncomputable def b := (2, -3) : ℝ × ℝ

def is_perpendicular_to (v w : ℝ × ℝ) : Prop := 
  let ⟨vx, vy⟩ := v
  let ⟨wx, wy⟩ := w
  vx * wx + vy * wy = 0

theorem find_k (k : ℝ) (h : is_perpendicular_to (k • a - 2 • b) a) : k = -1 := 
  sorry

end find_k_l239_239932


namespace coin_toss_probability_l239_239295

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  binomial_probability 3 2 0.5 = 0.375 :=
by
  sorry

end coin_toss_probability_l239_239295


namespace team_X_finishes_with_more_points_than_Y_l239_239015

-- Define the number of teams and games played
def numberOfTeams : ℕ := 8
def gamesPerTeam : ℕ := numberOfTeams - 1

-- Define the probability of winning (since each team has a 50% chance to win any game)
def probOfWin : ℝ := 0.5

-- Define the event that team X finishes with more points than team Y
noncomputable def probXFinishesMorePointsThanY : ℝ := 1 / 2

-- Statement to be proved: 
theorem team_X_finishes_with_more_points_than_Y :
  (∃ p : ℝ, p = probXFinishesMorePointsThanY) :=
sorry

end team_X_finishes_with_more_points_than_Y_l239_239015


namespace polygons_construction_l239_239836

noncomputable def number_of_polygons : ℝ := 15

theorem polygons_construction :
  (∑ n in finset.range (4 + 1) (number_of_polygons).to_nat, n * (n - 3) / 2) = 800 :=
sorry

end polygons_construction_l239_239836


namespace length_of_bridge_is_220_l239_239293

-- Define the given conditions
def train_length : ℝ := 155       -- in meters
def train_speed : ℝ := 45 * (1000 / 3600) -- converted to m/s
def crossing_time : ℝ := 30       -- in seconds

-- Calculate the total distance traveled by the train
def total_distance : ℝ := train_speed * crossing_time

-- Calculate the length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- Prove that the length of the bridge is 220 meters
theorem length_of_bridge_is_220 : bridge_length = 220 :=
  by
    unfold train_length
    unfold train_speed
    unfold crossing_time
    unfold total_distance
    unfold bridge_length
    sorry

end length_of_bridge_is_220_l239_239293


namespace solution_intervals_l239_239375

noncomputable def cubic_inequality (x : ℝ) : Prop :=
  x^3 - 3 * x^2 - 4 * x - 12 ≤ 0

noncomputable def linear_inequality (x : ℝ) : Prop :=
  2 * x + 6 > 0

theorem solution_intervals :
  { x : ℝ | cubic_inequality x ∧ linear_inequality x } = { x | -2 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solution_intervals_l239_239375


namespace boxes_needed_l239_239575

theorem boxes_needed (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 94) (h2 : boxes_capacity = 8) : 
  (total_oranges + boxes_capacity - 1) / boxes_capacity = 12 := 
by
  sorry

end boxes_needed_l239_239575


namespace fourier_transform_of_f_l239_239391
open Complex

-- Definitions for the problem conditions
def f (x : ℝ) (b : ℝ) : ℂ := Complex.exp (-b^2 * x^2)

-- Known Fourier transform pair
def known_fourier_pair (p : ℝ) : ℂ := (1 / Real.sqrt 2) * Complex.exp (-p^2 / 4)

-- Statement of the problem to prove
theorem fourier_transform_of_f (b : ℝ) (p : ℝ) :
  (Complex.fourier_transform (f x b)) p = (1 / (b * Real.sqrt 2)) * Complex.exp (-p^2 / (4 * b^2)) := sorry

end fourier_transform_of_f_l239_239391


namespace sec_sub_tan_equals_one_fifth_l239_239583

theorem sec_sub_tan_equals_one_fifth (x : ℝ) (h : Real.sec x + Real.tan x = 5) : Real.sec x - Real.tan x = 1 / 5 :=
by
  sorry

end sec_sub_tan_equals_one_fifth_l239_239583


namespace round_table_six_people_l239_239147

-- Definition of the problem
def round_table_arrangements (n : ℕ) : ℕ :=
  nat.factorial n / n

-- The main theorem statement
theorem round_table_six_people : round_table_arrangements 6 = 120 :=
by 
  -- We implement the definition and calculations inline here directly.
  sorry

end round_table_six_people_l239_239147


namespace eccentricity_of_tangent_hyperbola_l239_239127

theorem eccentricity_of_tangent_hyperbola :
  ∀ (C : Type) (hC : hyperbola_center_origin_and_foci_x_axis C)
  (tangent_condition : asymptotes_tangent_to_parabola C (λ y : ℝ, y^2 = x - 1)),
  eccentricity C = √5 / 2 := 
sorry

end eccentricity_of_tangent_hyperbola_l239_239127


namespace sin_double_angle_fourth_quadrant_l239_239955

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239955


namespace latest_time_to_start_roasting_turkeys_l239_239195

theorem latest_time_to_start_roasting_turkeys
  (turkeys : ℕ) 
  (weight_per_turkey : ℕ) 
  (minutes_per_pound : ℕ) 
  (dinner_time_hours : ℕ)
  (dinner_time_minutes : ℕ) 
  (one_at_a_time : turkeys = 2)
  (weight : weight_per_turkey = 16)
  (roasting_time_per_pound : minutes_per_pound = 15)
  (dinner_hours : dinner_time_hours = 18)
  (dinner_minutes : dinner_time_minutes = 0) :
  (latest_start_hours : ℕ) (latest_start_minutes : ℕ) :=
  latest_start_hours = 10 ∧ latest_start_minutes = 0 := 
sorry

end latest_time_to_start_roasting_turkeys_l239_239195


namespace diagonals_concurrent_l239_239421

structure Hexagon (ABCDEF : Polygon) :=
  (inscribed_circle : Circle)
  (circumcircle : Circle)
  (triangles_incircles : ∀ (t : {t // t ∈ triangles ABCDEF}), Circle)
  (ext_tangents : ∀ (i j : {t // t ∈ triangles ABCDEF}), i ≠ j → Line)
  (intersections : ∀ (i j : {t // t ∈ triangles ABCDEF}), i ≠ j → Point)

theorem diagonals_concurrent {ABCDEF : Polygon}
  (h : Hexagon ABCDEF)
  (convex : convex ABCDEF)
  (intersects : ∀ (i j : {t // t ∈ triangles h}), i ≠ j → configured_to_intersect i j)
  (convex_int : convex (create_int_hexagon h.ext_tangents h.intersections)) :
  diagonals_concurrent h.intersections := sorry

end diagonals_concurrent_l239_239421


namespace complement_of_intersection_l239_239191

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3}
def intersection : Set ℕ := M ∩ N
def complement : Set ℕ := U \ intersection

theorem complement_of_intersection (U M N : Set ℕ) :
  U = {0, 1, 2, 3} →
  M = {0, 1, 2} →
  N = {1, 2, 3} →
  (U \ (M ∩ N)) = {0, 3} := by
  intro hU hM hN
  simp [hU, hM, hN]
  sorry

end complement_of_intersection_l239_239191


namespace maximum_vertices_no_rectangle_l239_239735

theorem maximum_vertices_no_rectangle (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, m = 1009 ∧
  ∀ (V : Finset (Fin n)), V.card = m →
  ∀ (v1 v2 v3 v4 : Fin n), {v1, v2, v3, v4}.subset V →
  ¬ (v1.val + v3.val = v2.val + v4.val ∧ v1.val ≠ v2.val ∧ v1.val ≠ v3.val ∧ v1.val ≠ v4.val ∧ v2.val ≠ v3.val ∧ v2.val ≠ v4.val ∧ v3.val ≠ v4.val) :=
sorry

end maximum_vertices_no_rectangle_l239_239735


namespace reciprocal_of_repeating_decimal_6_l239_239749

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end reciprocal_of_repeating_decimal_6_l239_239749


namespace initial_quarters_l239_239619

variable (q : ℕ)

theorem initial_quarters (h : q + 3 = 11) : q = 8 :=
by
  sorry

end initial_quarters_l239_239619


namespace distance_from_center_to_plane_correct_l239_239896

noncomputable def distance_from_center_to_plane 
  (r : ℝ) (hp : ∀ (P A B C : EuclideanSpace ℝ (Fin 3)), 
                ∀ i j, i ≠ j → dotProduct (P - A) (P - B) = 0) : ℝ :=
let radius := sqrt 3 in
let P := λ i, EuclideanSpace.coordinates i P in
let A := λ i, EuclideanSpace.coordinates i A in
let B := λ i, EuclideanSpace.coordinates i B in
let C := λ i, EuclideanSpace.coordinates i C in
have hpa : ∀ i ≠ j, dotProduct (P - A) (P - B) = 0 := hp P A B C,
(r := radius) → (P, A, B, C lie on the sphere)
-- calculate OM
let OM := (sqrt 3) / 3 in
OM

-- Prove: distance_from_center_to_plane r hpa = (sqrt 3) / 3
theorem distance_from_center_to_plane_correct (r : ℝ) (hp : ∀ (P A B C : EuclideanSpace ℝ (Fin 3)), 
                                                   ∀ i j, i ≠ j → dotProduct (P - A) (P - B) = 0) :
  distance_from_center_to_plane r hp = (sqrt 3) / 3 := 
sorry

end distance_from_center_to_plane_correct_l239_239896


namespace sin_double_angle_fourth_quadrant_l239_239954

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239954


namespace impossible_difference_l239_239107

noncomputable def f (x : ℤ) (c1 c2 c3 c4 : ℤ) : ℤ :=
  (x^2 - 4 * x + c1) * (x^2 - 4 * x + c2) * (x^2 - 4 * x + c3) * (x^2 - 4 * x + c4)

def M (f : ℤ → ℤ) : Set ℤ := {x : ℤ | f(x) = 0}

theorem impossible_difference (c1 c2 c3 c4 : ℤ)
  (hc : c1 ≤ c2 ∧ c2 ≤ c3 ∧ c3 ≤ c4)
  (roots_in_M : ∃ (x1 x2 x3 x4 x5 x6 x7 : ℤ), M (f c1 c2 c3 c4) = {x1, x2, x3, x4, x5, x6, x7} ∧ ∀ xi, xi ∈ M (f c1 c2 c3 c4) → xi ∈ ℤ) :
  c4 - c1 ≠ 4 :=
by
  sorry

end impossible_difference_l239_239107


namespace part1_inequality_part2_range_of_a_l239_239532

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239532


namespace focal_length_of_hyperbola_l239_239489

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239489


namespace magnitude_BC_eq_sqrt29_l239_239419

noncomputable def A : (ℝ × ℝ) := (2, -1)
noncomputable def C : (ℝ × ℝ) := (0, 2)
noncomputable def AB : (ℝ × ℝ) := (3, 5)

theorem magnitude_BC_eq_sqrt29
    (A : ℝ × ℝ := (2, -1))
    (C : ℝ × ℝ := (0, 2))
    (AB : ℝ × ℝ := (3, 5)) :
    ∃ B : ℝ × ℝ, (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 29 := 
by
  sorry

end magnitude_BC_eq_sqrt29_l239_239419


namespace monthly_growth_rate_l239_239827

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end monthly_growth_rate_l239_239827


namespace cone_base_circumference_l239_239808

-- Definitions of the problem
def radius : ℝ := 5
def angle_sector_degree : ℝ := 120
def full_circle_degree : ℝ := 360

-- Proof statement
theorem cone_base_circumference 
  (r : ℝ) (angle_sector : ℝ) (full_angle : ℝ) 
  (h1 : r = radius) 
  (h2 : angle_sector = angle_sector_degree) 
  (h3 : full_angle = full_circle_degree) : 
  (angle_sector / full_angle) * (2 * π * r) = (10 * π) / 3 := 
by sorry

end cone_base_circumference_l239_239808


namespace find_ordered_pair_l239_239867

theorem find_ordered_pair : ∃ (x y : ℚ), 
  7 * x - 50 * y = -3 ∧ 3 * x - 2 * y = 8 ∧ 
  x = 599 / 204 ∧ y = 65 / 136 :=
by
  have h1 : 7 * (599 / 204) - 50 * (65 / 136) = -3 := sorry
  have h2 : 3 * (599 / 204) - 2 * (65 / 136) = 8 := sorry
  exact ⟨599 / 204, 65 / 136, h1, h2, rfl, rfl⟩

end find_ordered_pair_l239_239867


namespace sin_cos_alpha_sum_l239_239610

variables {a : ℝ} (h_a : a < 0)

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def P : ℝ × ℝ := (3 * a, -4 * a)
noncomputable def OP : ℝ := real.sqrt ((3 * a) ^ 2 + (-4 * a) ^ 2)

lemma OP_eq_5a : OP = (-5 * a) :=
begin
  rw [OP, real.sqrt_eq_rpow, pow_two, pow_two, add_mul, mul_assoc, pow_one],
  sorry -- proof steps
end

noncomputable def cos_alpha : ℝ := (3 * a) / OP
noncomputable def sin_alpha : ℝ := (-4 * a) / OP

lemma cos_alpha_value : cos_alpha = -(3 / 5) :=
begin
  rw [cos_alpha, OP_eq_5a],
  sorry -- proof steps
end

lemma sin_alpha_value : sin_alpha = 4 / 5 :=
begin
  rw [sin_alpha, OP_eq_5a],
  sorry -- proof steps
end

theorem sin_cos_alpha_sum : sin_alpha + cos_alpha = 1 / 5 :=
begin
  rw [sin_alpha_value, cos_alpha_value],
  sorry -- proof steps
end

end sin_cos_alpha_sum_l239_239610


namespace tetrahedron_midpoint_lines_perpendicular_l239_239216

theorem tetrahedron_midpoint_lines_perpendicular :
  ∀ (A B1 C D1 A1 B C1 D : Point) (H M : Point),
    -- Conditions
    is_regular_tetrahedron (Tetrahedron.mk A B1 C D1) ∧
    is_inscribed_in_cube (Cube.mk A B C D A1 B1 C1 D1) ∧
    is_intersection_diagonal_plane (Line.mk A C1) (Plane.mk B1 C D1) H ∧
    is_midpoint (Segment.mk A H) M ∧
    (dist C1 H) = (1 / 3) * (dist A H) -> 
    -- Conclusion
    are_pairwise_perpendicular (Line.mk M B1) (Line.mk M C) (Line.mk M D1) :=
sorry

end tetrahedron_midpoint_lines_perpendicular_l239_239216


namespace smallest_x_correct_l239_239009

noncomputable def smallest_x (K : ℤ) : ℤ := 135000

theorem smallest_x_correct (K : ℤ) :
  (∃ x : ℤ, 180 * x = K ^ 5 ∧ x > 0) → smallest_x K = 135000 :=
by
  sorry

end smallest_x_correct_l239_239009


namespace sin_double_angle_neg_l239_239950

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239950


namespace intersection_on_circumcircle_l239_239640

-- Defining a structure for a Triangle
structure Triangle (α : Type) [Field α] :=
(A B C : Point α)

-- Defining the external angle bisector and the perpendicular bisector
def external_angle_bisector {α : Type} [Field α] (T : Triangle α) : Line α :=
 sorry

def perpendicular_bisector {α : Type} [Field α] (T : Triangle α) : Line α := 
 sorry

-- Defining the circumcircle of the triangle
def circumcircle {α : Type} [Field α] (T : Triangle α) : Circle α := 
 sorry

-- Definition of the intersection point of two lines
def intersection_point {α : Type} [Field α] (l1 l2 : Line α) : Point α := 
 sorry

-- The main statement
theorem intersection_on_circumcircle 
  {α : Type} [Field α] 
  (T : Triangle α) : 
  let Ω := circumcircle T in
  let ext_angle_bis := external_angle_bisector T in
  let perp_bis := perpendicular_bisector T in
  let P := intersection_point ext_angle_bis perp_bis in
  P ∈ Ω := 
begin
  sorry,
end

end intersection_on_circumcircle_l239_239640


namespace soap_box_length_l239_239792

theorem soap_box_length : 
  ∃ L : ℕ, 
  let soap_box_volume := 60 * L in
  let carton_volume := 25 * 42 * 60 in
  (150 * soap_box_volume = carton_volume) → (L = 7) :=
begin
  sorry,
end

end soap_box_length_l239_239792


namespace domain_f_l239_239586

variable {α : Type} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def domain_f2x_minus_1 := set.Icc (-3 : α) 3

theorem domain_f (f : α → α) (h : set.Icc (-3 : α) 3 ⊆ set.preimage f (set.Icc (-3 : α) 3)) :
  set.Icc (-7 : α) 5 = set.preimage (λ x : α, 2 * x - 1) (set.Icc (-3 : α) 3) :=
by
  sorry

end domain_f_l239_239586


namespace f_f_3_eq_3_l239_239887

def f (x : ℝ) : ℝ :=
  if x < 3 then 3 * Real.exp (x - 1)
  else Real.logb 3 (x ^ 2 - 6)

theorem f_f_3_eq_3 : f (f 3) = 3 :=
  sorry

end f_f_3_eq_3_l239_239887


namespace trig_inequality_l239_239632

theorem trig_inequality :
  let a := Real.sin (55 * Real.pi / 180)
  let b := Real.cos (55 * Real.pi / 180)
  let c := Real.tan (55 * Real.pi / 180)
  in c > a ∧ a > b :=
by
  sorry

end trig_inequality_l239_239632


namespace focal_length_of_hyperbola_l239_239467

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239467


namespace percentage_shaded_in_square_l239_239282

theorem percentage_shaded_in_square
  (EFGH : Type)
  (square : EFGH → Prop)
  (side_length : EFGH → ℝ)
  (area : EFGH → ℝ)
  (shaded_area : EFGH → ℝ)
  (P : EFGH)
  (h_square : square P)
  (h_side_length : side_length P = 8)
  (h_area : area P = side_length P * side_length P)
  (h_small_shaded : shaded_area P = 4)
  (h_large_shaded : shaded_area P + 7 = 11) :
  (shaded_area P / area P) * 100 = 17.1875 :=
by
  sorry

end percentage_shaded_in_square_l239_239282


namespace monogram_count_l239_239648

/-- 
The number of monograms (first, middle, and last initials) possible
such that the initials are in alphabetical order with no letter repeated,
and the last initial is 'X', is 253.
-/
theorem monogram_count :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
  in (∃ (f: Fin 26 → Fin 26 → Fin 26 → 𝔹),
      (∀ (i j k : Fin 26), f i j k → alphabet.nth (i.val) < alphabet.nth (j.val) ∧ alphabet.nth (j.val) < alphabet.nth (k.val) ∧ alphabet.nth (k.val) = some 'X') 
      -> 253) :=
by
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
  have total_combinations := Nat.choose 23 2
  show total_combinations = 253
  sorry

end monogram_count_l239_239648


namespace relationship_among_S1_S2_S3_l239_239578

noncomputable def S1 : ℝ := ∫ (x : ℝ) in 1..2, x^2
noncomputable def S2 : ℝ := ∫ (x : ℝ) in 1..2, 1/x
noncomputable def S3 : ℝ := ∫ (x : ℝ) in 1..2, Real.exp x

theorem relationship_among_S1_S2_S3 : S2 < S1 ∧ S1 < S3 :=
by
  sorry

end relationship_among_S1_S2_S3_l239_239578


namespace barbara_shopping_l239_239348

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l239_239348


namespace number_of_subsets_including_123_l239_239842

theorem number_of_subsets_including_123 : 
  ∃ (X : Set (Set ℕ)), (X = {Y | {1, 2, 3} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6, 7}}) ∧ 
  (X.card = 2^4) :=
sorry

end number_of_subsets_including_123_l239_239842


namespace ratio_of_part_diminished_by_10_to_whole_number_l239_239797

theorem ratio_of_part_diminished_by_10_to_whole_number (N : ℝ) (x : ℝ) (h1 : 1/5 * N + 4 = x * N - 10) (h2 : N = 280) :
  x = 1 / 4 :=
by
  rw [h2] at h1
  sorry

end ratio_of_part_diminished_by_10_to_whole_number_l239_239797


namespace find_number_l239_239309

theorem find_number (x : ℝ) (h : 160 = 3.2 * x) : x = 50 :=
by 
  sorry

end find_number_l239_239309


namespace hyperbola_focal_length_l239_239463

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239463


namespace a_minus_3d_eq_zero_l239_239975

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x - 3 * d)

theorem a_minus_3d_eq_zero (a b c d : ℝ) (h : f a b c d ≠ 0)
  (h1 : ∀ x, f a b c d x = x) :
  a - 3 * d = 0 :=
sorry

end a_minus_3d_eq_zero_l239_239975


namespace find_105th_digit_l239_239132

theorem find_105th_digit :
  let sequence_digits := (List.range' 1 80).reverse.bind (fun n => toString n).data
  sequence_digits.get? 104 = some '2' :=
by
  sorry

end find_105th_digit_l239_239132


namespace inequality_abc_l239_239657

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) >= (a + b + c) / 3 := by
  sorry

end inequality_abc_l239_239657


namespace hyperbola_focal_length_l239_239461

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239461


namespace quadratic_root_q_value_l239_239911

theorem quadratic_root_q_value (p q : ℝ) (hr : is_complex_root 3 4 3 (-4) i p q) (h_real_coefs : p.is_real ∧ q.is_real) : q = 75 :=
    sorry

end quadratic_root_q_value_l239_239911


namespace num_of_integers_l239_239875

theorem num_of_integers (n : ℤ) (h : -1000 ≤ n ∧ n ≤ 1000) (h1 : 1 < 4 * n + 7) (h2 : 4 * n + 7 < 150) : 
  (∃ N : ℕ, N = 37) :=
by
  sorry

end num_of_integers_l239_239875


namespace last_locker_opened_l239_239809

theorem last_locker_opened (n : ℕ) (hn : n = 2048) :
  ∃ k, k = 2046 ∧ opens_last_locker k :=
begin
  sorry,
end

end last_locker_opened_l239_239809


namespace sum_series_eq_seven_twelve_l239_239364

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1) / ((n:ℝ) * (n + 1) * (n + 2) * (n + 3)) else 0

theorem sum_series_eq_seven_twelve : sum_series = 7 / 12 :=
by
  sorry

end sum_series_eq_seven_twelve_l239_239364


namespace folded_triangle_line_segment_length_squared_l239_239798

def equilateral_triangle (A B C : Point) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

def folded_line_segment_length_squared (A B C F : Point) (s d : ℝ) : ℝ :=
  sorry

theorem folded_triangle_line_segment_length_squared
  (A B C F : Point)
  (h_triangle : equilateral_triangle A B C 10)
  (h_fold : dist B F = 7)
  : folded_line_segment_length_squared A B C F 10 7 = 10405 / 881 :=
sorry

end folded_triangle_line_segment_length_squared_l239_239798


namespace provisions_last_days_l239_239775

def initial_men : ℕ := 1500
def initial_consumption : ℕ := 2
def additional_days_before_joining : ℕ := 10
def additional_men_high_metabolism : ℕ := 280
def high_metabolism_consumption : ℕ := 3
def additional_men_special_diet : ℕ := 40
def special_diet_consumption : ℕ := 1
def total_days_provisions_initial : ℕ := 17

theorem provisions_last_days :
  let remaining_food :=
        (initial_men * initial_consumption * total_days_provisions_initial)
        - (initial_men * initial_consumption * additional_days_before_joining)
  let total_food_consumption_per_day_after_joining :=
        (initial_men * initial_consumption)
        + (additional_men_high_metabolism * high_metabolism_consumption)
        + (additional_men_special_diet * special_diet_consumption)
  in remaining_food / total_food_consumption_per_day_after_joining = 5 := 
by
  sorry

end provisions_last_days_l239_239775


namespace focal_length_of_hyperbola_l239_239470

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239470


namespace correct_option_l239_239286

theorem correct_option (a b c : ℝ) : 
  (5 * a - (b + 2 * c) = 5 * a + b - 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a + b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b - 2 * c) ↔ 
  (5 * a - (b + 2 * c) = 5 * a - b - 2 * c) :=
by
  sorry

end correct_option_l239_239286


namespace correct_statements_l239_239012

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x, f x = x ^ a

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

def condition1 : Prop := is_power_function (λ _ => 1)

def condition2 (f : ℝ → ℝ) : Prop := is_odd_function f → f 0 = 0

def condition3 : Prop := is_odd_function (λ x => Real.log (x + Real.sqrt (x^2 + 1)))

def condition4 (a : ℝ) : Prop := a < 0 → (a^2)^(3/2) = a^3

def condition5 : Prop := ¬ ∃ x : ℝ, (λ _ => 1) x = 0

theorem correct_statements :
  (condition1 ∧ ∀ f, condition2 f ∧ condition3 ∧ ¬ ∀ a, condition4 a ∧ ¬ condition5) :=
by
  sorry

end correct_statements_l239_239012


namespace range_g_l239_239240

open Real

def floor (x : ℝ) : ℤ := ⌊x⌋

def g (x : ℝ) : ℝ := floor x - 2 * x

theorem range_g : (⋃ (n : ℤ), set.Ico (-(n : ℝ)) (-(n + 2))) = set.Iio 0 :=
by {
  -- placeholder for proof steps
  sorry
}

end range_g_l239_239240


namespace max_marked_vertices_no_rectangle_l239_239733

-- Definitions for the conditions
def regular_polygon (n : ℕ) := n ≥ 3

def no_four_marked_vertices_form_rectangle (n : ℕ) (marked_vertices : Finset ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : ℕ), 
  v1 ∈ marked_vertices ∧ 
  v2 ∈ marked_vertices ∧ 
  v3 ∈ marked_vertices ∧ 
  v4 ∈ marked_vertices → 
  (v1, v2, v3, v4) do not form the vertices of a rectangle in a regular n-gon

-- The theorem we need to prove
theorem max_marked_vertices_no_rectangle (marked_vertices : Finset ℕ) :
  marked_vertices.card ≤ 1009 :=
begin
  assume h1: regular_polygon 2016,
  assume h2: no_four_marked_vertices_form_rectangle 2016 marked_vertices,
  sorry
end

end max_marked_vertices_no_rectangle_l239_239733


namespace power_addition_l239_239414

variable {R : Type*} [CommRing R]

theorem power_addition (x : R) (m n : ℕ) (h1 : x^m = 6) (h2 : x^n = 2) : x^(m + n) = 12 :=
by
  sorry

end power_addition_l239_239414


namespace number_of_ways_l239_239698

/-- The number of ways eight people can line up to buy a ticket
if two of these people, Alice and Bob, insist on standing together is 10080. -/
theorem number_of_ways (n : ℕ) (A B : bool) (hN : n = 8) (hA : A = true) (hB : B = true) :
  ∃ k : ℕ, k = 7! * 2 ∧ k = 10080 :=
by
  sorry

end number_of_ways_l239_239698


namespace range_of_h_l239_239395

noncomputable def h (z : ℝ) : ℝ :=
  (z^2 + 5/2 * z + 2) / (z^2 + 2)

theorem range_of_h : 
  (set.range h) = set.Icc (3 / 4) (13 / 8) := 
sorry

end range_of_h_l239_239395


namespace largest_number_with_constraints_l239_239265

def is_valid_digit (d : ℕ) : Prop :=
d = 1 ∨ d = 2 ∨ d = 3

def digits_sum_to (n : ℕ) (digits : List ℕ) : Prop :=
digits.sum = n

def is_largest_number (digits : List ℕ) : Prop :=
digits = [2, 2, 2, 2, 1, 1, 1, 1, 1]

theorem largest_number_with_constraints : 
  ∃ digits : List ℕ, 
    (∀ d ∈ digits, is_valid_digit d) ∧
    digits_sum_to 13 digits ∧
    is_largest_number digits :=
begin
  sorry
end

end largest_number_with_constraints_l239_239265


namespace smallest_solution_l239_239059

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239059


namespace find_integer_n_l239_239263

theorem find_integer_n :
  ∃ n : ℤ, 0 ≤ n ∧ n < 17 ∧ (-150 ≡ n [MOD 17]) ∧ n ∣ 102 :=
sorry

end find_integer_n_l239_239263


namespace inequality_holds_l239_239839

theorem inequality_holds (a b : ℝ) (h : 0 < a ∧ a ≤ 4 ∧ b = 9 - a) : 
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 → 
  (x * y + y * z + z * x) ≥ a * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + b * (x * y * z) :=
by {
  intro x y z,
  intro hxyz,
  sorry
}

end inequality_holds_l239_239839


namespace smallest_solution_l239_239041

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239041


namespace barbara_spent_on_other_goods_l239_239355

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l239_239355


namespace john_has_leftover_correct_l239_239620

-- Define the initial conditions
def initial_gallons : ℚ := 5
def given_away : ℚ := 18 / 7

-- Define the target result after subtraction
def remaining_gallons : ℚ := 17 / 7

-- The theorem statement
theorem john_has_leftover_correct :
  initial_gallons - given_away = remaining_gallons :=
by
  sorry

end john_has_leftover_correct_l239_239620


namespace number_of_valid_n_l239_239634

theorem number_of_valid_n :
  (∃! n : ℕ, n ≥ 3 ∧
    (∀ (z : Fin n → ℂ), (∀ i, ∥z i∥ = 1) → (∑ i, z i ^ 3 = 0) →
      (∃ i, z i = 1) ∧ ∀ j k, z j = z k ∨ ∠ (z j, 1) = ∠ (z k, 1) + (k - j) * (2 * Real.pi / n)
    )) :=
sorry

end number_of_valid_n_l239_239634


namespace focal_length_of_hyperbola_l239_239484

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239484


namespace distance_sum_eqn_l239_239500

def dist_point_to_line (x1 y1 A B C : ℝ) : ℝ :=
  abs (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2)

theorem distance_sum_eqn :
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let A := 1
  let B := -1
  let C := -2
  let l_eq := (1, -1, -2) -- Coefficients of x - y - 2 = 0 
  dist_point_to_line F1.1 F1.2 A B C + dist_point_to_line F2.1 F2.2 A B C = 2 * real.sqrt 2 :=
by
  -- Proof here
  sorry

end distance_sum_eqn_l239_239500


namespace monthly_growth_rate_l239_239825

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l239_239825


namespace max_sin_A_l239_239136

variable {V : Type} [InnerProductSpace ℝ V]

/-- Proof problem translated from the given solution to Lean -/
theorem max_sin_A (A B C : V)
  (m : V) (h1 : m = (C - B) - 2 * (C - A))
  (n : V) (h2 : n = (B - A) - (C - A))
  (h_perp : ⟪m, n⟫ = 0) : 
  ∃ θ, θ = Real.sin (angle B A C) ∧ θ ≤ 1 / 2 :=
by
  sorry

end max_sin_A_l239_239136


namespace probability_of_6_largest_l239_239315

-- Problem statement and conditions
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def number_of_draws : ℕ := 4

-- Define a function to simulate the card drawing without replacement
noncomputable def draw_cards (c : List ℕ) (n : ℕ) : List (List ℕ) :=
List.powersetLen n c

-- Define the event that 6 is the largest card in the draw
def event_6_largest (draw : List ℕ) : Prop :=
draw.maximum' = 6

-- Define the probability measure for the desired event
noncomputable def probability_6_largest : ℚ :=
let all_draws := draw_cards cards number_of_draws in
let favorable_draws := all_draws.filter event_6_largest in
(favorable_draws.length : ℚ) / (all_draws.length : ℚ)

-- Proof statement
theorem probability_of_6_largest : probability_6_largest = 2 / 7 :=
sorry

end probability_of_6_largest_l239_239315


namespace minPathSum_correct_maxPathSum_correct_l239_239769

section MinMaxPathSum

variables (n : ℕ)

/-- The minimum possible path sum in an n x n multiplication table from top-left to bottom-right -/
def minPathSum : ℕ :=
  (n * (n^2 + 2 * n - 1) / 2)

theorem minPathSum_correct (n : ℕ) : 
  ∃ p, is_valid_path p n ∧ path_sum p = (n * (n^2 + 2n - 1) / 2) :=
  sorry

/-- The maximum possible path sum in an n x n multiplication table from top-left to bottom-right -/
def maxPathSum : ℕ :=
  (n * (n + 1) * (4n - 1) / 6)

theorem maxPathSum_correct (n : ℕ) : 
  ∃ p, is_valid_path p n ∧ path_sum p = (n * (n + 1) * (4n - 1) / 6) :=
  sorry

/-- Predicate describing a valid path -/
def is_valid_path (p : List (ℕ × ℕ)) (n : ℕ) : Prop :=
  sorry

/-- Function to calculate the sum of the values in the multiplication table along a given path -/
def path_sum (p : List (ℕ × ℕ)) : ℕ :=
  sorry

end MinMaxPathSum

end minPathSum_correct_maxPathSum_correct_l239_239769


namespace sin_double_angle_fourth_quadrant_l239_239973

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239973


namespace square_condition_diagonal_length_l239_239114

-- Problem 1: The value of m for which ABCD is a square
theorem square_condition (m : ℝ) :
  (∀ (x : ℝ), x^2 - m * x + m / 2 - 1 / 4 = 0 → m = 1) :=
sorry

-- Problem 2: The length of the diagonal when AB = 2
theorem diagonal_length (m t : ℝ) (h1: 2 + t = m) (h2: 2 * t = m / 2 - 1 / 4) :
  sqrt (2^2 + t^2) = sqrt 17 / 2 :=
sorry

end square_condition_diagonal_length_l239_239114


namespace circle_radius_center_l239_239682

theorem circle_radius_center (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y - 2 = 0) :
  (∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = 1 ∧ r = 2) := 
sorry

end circle_radius_center_l239_239682


namespace part1_solution_part2_solution_l239_239554

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239554


namespace emily_prob_3_spaces_away_l239_239852

def is_equivalent_3_spaces_away (initial : Int) (spins : List Int) : Prop :=
  match spins with
  | [spin1, spin2] => (spin1 + spin2 = 3 ∨ spin1 + spin2 = -3)
  | _ => False

def probability_of_3_spaces_away : ProbabilitySpace (List Int) := sorry

theorem emily_prob_3_spaces_away :
  (probability_of_3_spaces_away {spins | is_equivalent_3_spaces_away initial spins}) = 7 / 16 := 
sorry

end emily_prob_3_spaces_away_l239_239852


namespace brad_made_two_gallons_of_lemonade_l239_239828

noncomputable def brad_profit_problem : ℕ :=
  let G := 2 in
  let glasses_per_gallon := 16 in
  let cost_per_gallon := 3.50 in
  let price_per_glass := 1.00 in
  let consumed_glasses := 5 in
  let unsold_glasses := 6 in
  let net_profit := 14 in
  (16 * G - consumed_glasses - unsold_glasses) * price_per_glass - cost_per_gallon * G = net_profit

theorem brad_made_two_gallons_of_lemonade : brad_profit_problem = 2 :=
sorry

end brad_made_two_gallons_of_lemonade_l239_239828


namespace beth_jan_total_money_l239_239981

theorem beth_jan_total_money (beth_money jan_money : ℕ)
    (h1 : beth_money + 35 = 105)
    (h2 : jan_money - 10 = beth_money) : beth_money + jan_money = 150 :=
begin
  sorry
end

end beth_jan_total_money_l239_239981


namespace monthly_growth_rate_l239_239824

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l239_239824


namespace sailboat_speed_max_power_l239_239689

-- Define constants for the problem.
def B : ℝ := sorry -- Aerodynamic force coefficient (to be provided)
def ρ : ℝ := sorry -- Air density (to be provided)
def S : ℝ := 7 -- sail area in m²
def v0 : ℝ := 6.3 -- wind speed in m/s

-- Define the force formula
def F (v : ℝ) : ℝ := (B * S * ρ * (v0 - v)^2) / 2

-- Define the power formula
def N (v : ℝ) : ℝ := F v * v

-- Define the condition that the power reaches its maximum value at some speed
def N_max : ℝ := sorry -- maximum instantaneous power (to be provided)

-- The proof statement that the speed of the sailboat when the power is maximized is v0 / 3
theorem sailboat_speed_max_power : ∃ v : ℝ, (N v = N_max ∧ v = v0 / 3) := 
  sorry

end sailboat_speed_max_power_l239_239689


namespace limit_of_abs_convergent_series_l239_239225

open Filter

noncomputable def abs_convergent_series (a : ℕ → ℕ → ℝ) (i : ℕ) : Prop :=
  summable (λ j, |a i j|)

theorem limit_of_abs_convergent_series
  (a : ℕ → ℕ → ℝ)
  (h_summable : ∀ i, abs_convergent_series a i)
  (h_limit : ∀ b : ℕ → ℝ, bounded b → tendsto (λ i, ∑' j, a i j * b j) at_top (nhds 0)) :
  tendsto (λ i, ∑' j, |a i j|) at_top (nhds 0) := sorry

end limit_of_abs_convergent_series_l239_239225


namespace person_can_pass_l239_239804

def radius_of_earth : ℝ := 6 * 10^6  -- approximated radius of the Earth in cm

def lengthened_rope_circumference (R : ℝ) : ℝ :=
  2 * Real.pi * R + 1 -- rope lengthened by 1 cm

def height_of_gap (R : ℝ) : ℝ :=
  (lengthened_rope_circumference R / (2 * Real.pi)) - R

theorem person_can_pass (R : ℝ) (h : ℝ) :
  R = radius_of_earth →
  h = height_of_gap R →
  h ≈ 7 :=
by
  intros
  sorry

end person_can_pass_l239_239804


namespace distribution_methods_l239_239777

theorem distribution_methods (volunteers schools : ℕ) (h1 : volunteers = 75) (h2 : schools = 3) (h3 : schools > 0) : 
  ∃ n, n = 150 ∧ school_distribution volunteers schools ≥ 1 := 
sorry

end distribution_methods_l239_239777


namespace maximum_vertices_no_rectangle_l239_239737

theorem maximum_vertices_no_rectangle (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, m = 1009 ∧
  ∀ (V : Finset (Fin n)), V.card = m →
  ∀ (v1 v2 v3 v4 : Fin n), {v1, v2, v3, v4}.subset V →
  ¬ (v1.val + v3.val = v2.val + v4.val ∧ v1.val ≠ v2.val ∧ v1.val ≠ v3.val ∧ v1.val ≠ v4.val ∧ v2.val ≠ v3.val ∧ v2.val ≠ v4.val ∧ v3.val ≠ v4.val) :=
sorry

end maximum_vertices_no_rectangle_l239_239737


namespace decrease_in_demand_correct_l239_239822

noncomputable def proportionate_decrease_in_demand (p e : ℝ) : ℝ :=
  1 - (1 / (1 + e * p))

theorem decrease_in_demand_correct :
  proportionate_decrease_in_demand 0.20 1.5 = 0.23077 :=
by
  sorry

end decrease_in_demand_correct_l239_239822


namespace convex_polygon_sides_ne_14_l239_239330

noncomputable def side_length : ℝ := 1

def is_triangle (s : ℝ) : Prop :=
  s = side_length

def is_dodecagon (s : ℝ) : Prop :=
  s = side_length

def side_coincide (t : ℝ) (d : ℝ) : Prop :=
  is_triangle t ∧ is_dodecagon d ∧ t = d

def valid_resulting_sides (s : ℤ) : Prop :=
  s = 11 ∨ s = 12 ∨ s = 13

theorem convex_polygon_sides_ne_14 : ∀ t d, side_coincide t d → ¬ valid_resulting_sides 14 := 
by
  intro t d h
  sorry

end convex_polygon_sides_ne_14_l239_239330


namespace smallest_solution_l239_239036

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239036


namespace find_m_plus_n_plus_d_l239_239140
noncomputable section

-- Define the problem statement
def circle_radius := 36
def chord_length := 66
def intersection_distance := 12
def result_m := 294
def result_n := 81
def result_d := 3

theorem find_m_plus_n_plus_d :
  ∃ (m n d : ℕ), 
    (m * (π : ℝ) - n * real.sqrt d) = 
    (294 * (π : ℝ) - 81 * real.sqrt 3) ∧ 
    m + n + d = 378 :=
by 
  use [result_m, result_n, result_d]
  split
  { rw [real.sqrt_eq_rsqrt, real.sqrt_eq_rsqrt],
    exact rfl, -- establish correct area formula equivalence
    },
  { norm_num, -- establish the sum m + n + d
    }

end find_m_plus_n_plus_d_l239_239140


namespace number_of_possible_values_l239_239243

theorem number_of_possible_values (x : ℕ) (h1 : x > 6) (h2 : x + 4 > 0) :
  ∃ (n : ℕ), n = 24 := 
sorry

end number_of_possible_values_l239_239243


namespace intersection_of_A_and_B_l239_239925

noncomputable def A : Set ℝ := {x | x^2 - 1 ≤ 0}

noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l239_239925


namespace part1_part2_l239_239545

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239545


namespace divisible_numbers_l239_239871

theorem divisible_numbers (n : ℕ) (k : ℕ) (h_k : 2 ≤ k) (lst : list ℤ) (h_len : lst.length = n) :
  ∃ steps : list (list ℤ → list ℤ), (∀ (f : list ℤ → list ℤ), f ∈ steps → ∃ i j, 0 ≤ i ∧ i ≤ j ∧ j < n ∧
        (∀ m, i ≤ m ∧ m ≤ j → (f lst).nth m = (lst.nth m).map (λ x, x + 1) ∨ 
                                 (f lst).nth m = (lst.nth m).map (λ x, x - 1))) →
    ∃ (final_state : list ℤ), final_state = (steps.foldl (flip ($)) lst) ∧
    (final_state.filter (λ x, x % k = 0)).length ≥ n - k + 2 :=
sorry

end divisible_numbers_l239_239871


namespace derangement_count_l239_239799

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def subfactorial (n : ℕ) : ℕ :=
  nat.factorial n * 
    (finset.sum (finset.range (n + 1)) 
      (λ k, (-1:ℤ)^k / (nat.factorial k : ℤ)))

theorem derangement_count (n : ℕ) : 
  ∃ (d : ℕ), d = subfactorial n :=
by
  sorry

end derangement_count_l239_239799


namespace isosceles_triangle_angle_bisector_l239_239835

theorem isosceles_triangle_angle_bisector (A B C X : Point) (h1 : dist A B = 80)
  (h2 : dist A C = 80) (h3 : dist B C = 100) (hX : AngleBisector A B C X) :
  dist A X = 80 :=
sorry

end isosceles_triangle_angle_bisector_l239_239835


namespace tea_per_cup_approx_l239_239720

-- Definitions of given conditions
def totalTeaMl : ℕ := 1050
def numberOfCups : ℕ := 16

-- Theorem stating the expected result
theorem tea_per_cup_approx :
  totalTeaMl / numberOfCups ≈ 66 :=
sorry

end tea_per_cup_approx_l239_239720


namespace surface_area_implies_side_length_diagonal_l239_239751

noncomputable def cube_side_length_diagonal (A : ℝ) := 
  A = 864 → ∃ s d : ℝ, s = 12 ∧ d = 12 * Real.sqrt 3

theorem surface_area_implies_side_length_diagonal : 
  cube_side_length_diagonal 864 := by
  sorry

end surface_area_implies_side_length_diagonal_l239_239751


namespace sum_of_three_numbers_l239_239700

theorem sum_of_three_numbers :
  ∃ (a b c : ℕ), 
    (a ≤ b ∧ b ≤ c) ∧ 
    (b = 8) ∧ 
    ((a + b + c) / 3 = a + 8) ∧ 
    ((a + b + c) / 3 = c - 20) ∧ 
    (a + b + c = 60) :=
by
  sorry

end sum_of_three_numbers_l239_239700


namespace option_C_correct_l239_239515

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem option_C_correct (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) (hl : ∀ x, ln x > -1) : 
  x₁ * f x₁ + x₂ * f x₂ > 2 * x₂ * f x₁ := 
sorry

end option_C_correct_l239_239515


namespace range_of_j_l239_239183

def h (x: ℝ) : ℝ := 2 * x + 1
def j (x: ℝ) : ℝ := h (h (h (h (h x))))

theorem range_of_j :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -1 ≤ j x ∧ j x ≤ 127 :=
by 
  intros x hx
  sorry

end range_of_j_l239_239183


namespace horner_evaluation_of_f_at_5_l239_239411

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_evaluation_of_f_at_5 : f 5 = 2015 :=
by sorry

end horner_evaluation_of_f_at_5_l239_239411


namespace octagon_opposite_sides_equal_l239_239818

theorem octagon_opposite_sides_equal
    (a b c d e f g h : ℕ)
    (equal_angles : ∀ i j, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 → internal_angle i = 135)
    (is_integer_side_lengths : ∀ i, 1 ≤ i ∧ i ≤ 8 → side_length i ∈ ℤ) :
  a = e ∧ b = f ∧ c = g ∧ d = h := 
sorry

end octagon_opposite_sides_equal_l239_239818


namespace find_a_l239_239504

theorem find_a (a : ℝ) (h : ∀ x, (deriv (λ x : ℝ, x^3 + a * x) 1 = 2)) : a = -1 :=
by
  simp only [deriv, differentiable_at_id', deriv_pow, deriv_mul, deriv_id', deriv_const, mul_one,
             mul_comm, one_pow, add_comm] at h
  have h' : 3 + a = 2 := by exact h 1
  linarith only [h']

end find_a_l239_239504


namespace ashwin_rental_hours_l239_239294

theorem ashwin_rental_hours (x : ℕ) 
  (h1 : 25 + 10 * x = 125) : 1 + x = 11 :=
by
  sorry

end ashwin_rental_hours_l239_239294


namespace cos_sum_identity_l239_239906

theorem cos_sum_identity (α : ℝ) (h_cos : Real.cos α = 3 / 5) (h_alpha : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (α + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 :=
by
  sorry

end cos_sum_identity_l239_239906


namespace smallest_solution_l239_239043

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239043


namespace mark_score_ratio_l239_239141

theorem mark_score_ratio (highest_score : ℕ) (range : ℕ) (mark_score : ℕ) (least_score : ℕ)
  (h1 : highest_score = 98)
  (h2 : range = 75)
  (h3 : mark_score = 46)
  (h4 : mark_score = least_score * 2) : 
  (mark_score : ℚ) / least_score = 2 := 
by
  have h5 : least_score = highest_score - range := by
    rw [h1, h2]
    exact rfl
  have h6 : least_score = 23 := by
    rw [h5, h1, h2]
    exact rfl
  rw [h3, h6]
  exact rfl

end mark_score_ratio_l239_239141


namespace sum_of_x_coordinates_l239_239398

-- Definitions for the equations given in the conditions
def eq1 (x : ℝ) : ℝ := |x^2 - 8 * x + 12|
def eq2 (x : ℝ) : ℝ := 4 - x

-- Final problem statement translating our conditions to the proposition to be proved
theorem sum_of_x_coordinates : 
  (let coords := {x : ℝ | eq1 x = eq2 x} in 
  ∑ coord in coords, coord) = 16 :=
sorry

end sum_of_x_coordinates_l239_239398


namespace real_roots_system_l239_239008

theorem real_roots_system :
  ∃ (x y : ℝ), 
    (x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97) ↔ 
    (x, y) = (3, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (-3, -2) ∨ (x, y) = (-2, -3) := 
by 
  sorry

end real_roots_system_l239_239008


namespace max_value_of_f_l239_239699

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 1 then 1 / x else - x ^ 2 + 2

theorem max_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 2 :=
by
  use 0  -- Let's start by using x = 0
  sorry  -- Placeholder for the proof

end max_value_of_f_l239_239699


namespace circle_radius_l239_239783

theorem circle_radius (O A B : Type) (a b : ℝ) (h₁ : dist O A = a) (h₂ : dist O B = b) :
  -- Prove the radius of the circle touching one side of the right angle at O
  -- and intersecting the other side at points A and B is (a + b) / 2.
  sorry

end circle_radius_l239_239783


namespace solution_set_empty_l239_239072

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l239_239072


namespace work_completed_in_approx_6_15_days_l239_239303

theorem work_completed_in_approx_6_15_days (x y z : ℕ → ℝ) :
  (∀ t : ℕ, x (t + 1) = x t + 1/20 ∧ y (t + 1) = y t + (3/80) ∧ z (t + 1) = z t + (3/40)) →
  ∃ d : ℝ, d ≈ 6.15 ∧ ∀ t : ℕ, x t + y t + z t = d :=
begin
  sorry
end

end work_completed_in_approx_6_15_days_l239_239303


namespace fraction_of_earth_surface_inhabitable_l239_239130

theorem fraction_of_earth_surface_inhabitable (f_land : ℚ) (f_inhabitable_land : ℚ)
  (h1 : f_land = 1 / 3)
  (h2 : f_inhabitable_land = 2 / 3) :
  f_land * f_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_of_earth_surface_inhabitable_l239_239130


namespace smallest_area_of_right_triangle_l239_239752

theorem smallest_area_of_right_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8) : 
  ∃ c : ℝ, smallest_area_of_triangle a b c ∧ c = 8 ∧ 
  (triangle_area 7 (Real.sqrt 15) = (7 * (Real.sqrt 15)) / 2) := by
  sorry

end smallest_area_of_right_triangle_l239_239752


namespace coprime_product_consecutive_integers_l239_239306

theorem coprime_product_consecutive_integers :
  ∀ n : ℕ, n ≥ 1 → ∃ k : fin (n+1) → ℕ, (∀ i : fin (n+1), k i > 1) ∧
  (∀ i j : fin (n+1), i ≠ j → Nat.coprime (k i) (k j)) ∧
  ∃ a b : ℕ, a = b + 1 ∧ (∏ i, k i) - 1 = a * b := by
  sorry

end coprime_product_consecutive_integers_l239_239306


namespace volume_set_S_l239_239624

open Real EuclideanSpace 

namespace VolumeOfS

def dist_line (X : ℝ³) : ℝ :=
  let (x, y, z) := X
  sqrt ((x + d)^2 + y^2)

def dist_point (X : ℝ³) : ℝ :=
  let (x, y, z) := X
  sqrt (x^2 + y^2 + z^2)

def set_S (X : ℝ³) (P : ℝ³) (ℓ : Set ℝ³) : Set ℝ³ :=
  { X | dist_line X ≥ 2 * dist_point X }

theorem volume_set_S (d : ℝ) (h_d_pos : d > 0) (ℓ : Set ℝ³) (P : ℝ³) : 
  ∀ P, ℓ, volume (set_S P ℓ) = 16 * π * d^3 / (27 * sqrt 3) :=
sorry

end VolumeOfS

end volume_set_S_l239_239624


namespace division_of_fractions_l239_239577

theorem division_of_fractions : (1 / 6) / (1 / 3) = 1 / 2 :=
by
  sorry

end division_of_fractions_l239_239577


namespace magnitude_of_z_not_2_z_times_conjugate_power_of_z_minus_1_not_minus_1_root_of_polynomial_l239_239420

namespace ComplexNumberExample

open Complex

-- Define the complex number z
def z : ℂ := 1 + I

-- theorem to prove the magnitude is not equal to 2
theorem magnitude_of_z_not_2 : abs z ≠ 2 := by
  sorry

-- theorem to prove z times its conjugate equals 2
theorem z_times_conjugate : z * conj z = 2 := by
  sorry

-- theorem to prove (z-1) to the power 2024 is not -1
theorem power_of_z_minus_1_not_minus_1 : (z - 1) ^ 2024 ≠ -1 := by
  sorry

-- theorem to prove a equals to 2 given the condition
theorem root_of_polynomial (a : ℝ) : (z: ℂ) is_one_root_of (a : ℝ) ↔ a = 2 := 
begin
  sorry
end

end ComplexNumberExample

end magnitude_of_z_not_2_z_times_conjugate_power_of_z_minus_1_not_minus_1_root_of_polynomial_l239_239420


namespace round_table_six_people_l239_239148

-- Definition of the problem
def round_table_arrangements (n : ℕ) : ℕ :=
  nat.factorial n / n

-- The main theorem statement
theorem round_table_six_people : round_table_arrangements 6 = 120 :=
by 
  -- We implement the definition and calculations inline here directly.
  sorry

end round_table_six_people_l239_239148


namespace base_4_conversion_odd_digit_count_350_l239_239393

-- Definition of base-4 conversion
def to_base4 (n : ℕ) : List ℕ :=
  let rec convert (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 4) :: convert (n / 4)
  convert n |>.reverse

-- Definition of counting odd digits in a list
def count_odd_digits (lst : List ℕ) : ℕ :=
  lst.filter (fun x => x % 2 == 1) |>.length

-- Main theorem statement
theorem base_4_conversion_odd_digit_count_350 :
  count_odd_digits (to_base4 350) = 4 :=
by
  sorry

end base_4_conversion_odd_digit_count_350_l239_239393


namespace find_complex_number_l239_239025

theorem find_complex_number (z : ℂ) (h : z * (1 + complex.i) + complex.i = 0) : z = -1 / 2 - (1 / 2) * complex.i :=
by
  sorry

end find_complex_number_l239_239025


namespace focal_length_of_hyperbola_l239_239474

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239474


namespace barbara_spent_on_other_goods_l239_239354

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l239_239354


namespace comparison_of_a_and_c_l239_239702

variable {α : Type _} [LinearOrderedField α]

theorem comparison_of_a_and_c (a b c : α) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end comparison_of_a_and_c_l239_239702


namespace pyramid_circumscribed_sphere_surface_area_l239_239227

theorem pyramid_circumscribed_sphere_surface_area :
  ∀ (l w h : ℝ), l = 7 → w = 5 → h = 8 →
  let R := (Real.sqrt (l^2 + w^2 + h^2)) / 2 in
  let S := 4 * Real.pi * R^2 in
  S = 138 * Real.pi :=
by
  intros l w h hl hw hh R S
  rw [hl, hw, hh]
  simp only [Real.sqrt_eq_rpow, Real.pi]
  sorry

end pyramid_circumscribed_sphere_surface_area_l239_239227


namespace find_other_root_l239_239106

theorem find_other_root (a b c : ℝ) :
  ∃ r₂, (λ x, x^2 - (a + b + c) * x + (a * b + b * c + c * a)) 2 = 0 ∧ r₂ = a + b + c - 2 :=
by
  sorry

end find_other_root_l239_239106


namespace arrangement_of_students_l239_239006

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_of_students : 
  let total_students := 6 in
  let students_in_jinan_min := 2 in
  let students_in_qingdao_min := 3 in
  let students_in_jinan_max := total_students - students_in_qingdao_min in
  let students_in_qingdao_max := total_students - students_in_jinan_min in
  students_in_qingdao_min <= total_students ∧ students_in_jinan_min <= total_students →
  binomial_coefficient total_students (total_students - students_in_qingdao_min) + 
  binomial_coefficient total_students (total_students - students_in_jinan_max) = 35 := 
by
  sorry

end arrangement_of_students_l239_239006


namespace sqrt_eq_sum_iff_conditions_l239_239846

theorem sqrt_eq_sum_iff_conditions (a b c : ℝ) :
  sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a + b + c ≥ 0 ∧ a * b + a * c + b * c = 0 :=
by
  sorry

end sqrt_eq_sum_iff_conditions_l239_239846


namespace distance_is_correct_l239_239760

noncomputable def distance_from_home_to_forest_park : ℝ := 11  -- distance in kilometers

structure ProblemData where
  v : ℝ                  -- Xiao Wu's bicycling speed (in meters per minute)
  t_catch_up : ℝ          -- time it takes for father to catch up (in minutes)
  d_forest : ℝ            -- distance from catch-up point to forest park (in kilometers)
  t_remaining : ℝ        -- time remaining for Wu to reach park after wallet delivered (in minutes)
  bike_speed_factor : ℝ   -- speed factor of father's car compared to Wu's bike
  
open ProblemData

def problem_conditions : ProblemData :=
  { v := 350,
    t_catch_up := 7.5,
    d_forest := 3.5,
    t_remaining := 10,
    bike_speed_factor := 5 }

theorem distance_is_correct (data : ProblemData) :
  data.v = 350 →
  data.t_catch_up = 7.5 →
  data.d_forest = 3.5 →
  data.t_remaining = 10 →
  data.bike_speed_factor = 5 →
  distance_from_home_to_forest_park = 11 := 
by
  intros
  sorry

end distance_is_correct_l239_239760


namespace fixed_point_2_5_l239_239242

def fixed_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : Prop :=
  (2, 5) ∈ {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ y = a^(x - 2) + 4}

theorem fixed_point_2_5 (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : fixed_point a h h1 :=
begin
  dsimp [fixed_point],
  use 2, 5,
  exact sorry,
end

end fixed_point_2_5_l239_239242


namespace distribute_students_l239_239378

theorem distribute_students :
  let students := {A, B, C, D}
  let classes := {class1, class2, class3}
  ∃ (distribution : students → classes), 
  (∀ s, distribution s ∈ classes) ∧
  ((∀ cls, ∃ sth ∈ students, distribution sth = cls) ∧ 
  (distribution A ≠ distribution B) ∧ distribution_set.nodup)
  → 
  distribution_set.distinct_count = 30 :=

sorry

end distribute_students_l239_239378


namespace cesaro_sum_of_100_terms_l239_239502

theorem cesaro_sum_of_100_terms (a : ℕ → ℕ) 
    (h : (∑ i in range 99, (99 - i) * a (i + 1)) / 99 = 1000) : 
    (∑ i in range 100, ((∑ j in range (i + 1), if j = 0 then 2 else a j) )) / 100 = 992 := 
sorry

end cesaro_sum_of_100_terms_l239_239502


namespace equal_distances_l239_239093

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) := { p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1 }

theorem equal_distances (a b : ℝ) (h : a > b ∧ b > 0) (h_focal : 2 * real.sqrt (a ^ 2 - b ^ 2) = 4)
  (F : ℝ × ℝ) (hF : F = (-2, 0)) (P : ℝ × ℝ) (hP : ∃ m : ℝ, P = (-3, m))
  (d1 d2 : ℝ) (M N : ℝ × ℝ)
  (H_MN : ∃ k : ℝ, k ≠ 0 ∧ ∀ (x : ℝ × ℝ), (x ∈ ellipse a b h) → 
    p.1 = k * p.2 - 2 → x = M ∨ x = N)
  (O : ℝ × ℝ := (0, 0))
  (d_1_dist : distance M O = d1)
  (d_2_dist : distance N O = d2) :
  d1 = d2
  :=
sorry

end equal_distances_l239_239093


namespace increase_in_tire_radius_l239_239014

theorem increase_in_tire_radius
  (r : ℝ)
  (d1 d2 : ℝ)
  (conv_factor : ℝ)
  (original_radius : r = 16)
  (odometer_reading_outbound : d1 = 500)
  (odometer_reading_return : d2 = 485)
  (conversion_factor : conv_factor = 63360) :
  ∃ Δr : ℝ, Δr = 0.33 :=
by
  sorry

end increase_in_tire_radius_l239_239014


namespace sin_of_30_deg_l239_239903

theorem sin_of_30_deg: 
  let α : ℝ := (30 * Real.pi) / 180 in
  Real.sin α = 1/2 := by
  sorry

end sin_of_30_deg_l239_239903


namespace problem1_problem2_l239_239438

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

theorem problem1 :
  ∀ (P : ℝ × ℝ)
  (A : ℝ × ℝ) (hx : A = (2, 0))
  (h : ∃ (x y : ℝ), P = (x, y) ∧ (sqrt ((x + 2)^2 + y^2) + sqrt ((x - 2)^2 + y^2) = 4 * sqrt 2)),
  trajectory_equation P.1 P.2 :=
sorry

theorem problem2 :
  ∀ (k : ℝ),
  k ≠ 0 →
  let A : ℝ × ℝ := (2, 0),
      l : ℝ × ℝ → ℝ := λ P, k * (P.1 - 2),
      D : ℝ × ℝ := (2 * k^2 / (1 + 2 * k^2), 0),
      MN : ℝ × ℝ := (4 * k^2 / (1 + 2 * k^2), -2 * k / (1 + 2 * k^2)),
      H : ℝ × ℝ := (2 * k^2 / (1 + 2 * k^2), -k / (1 + 2 * k^2)) in
  (0 < ∥D - H∥ / ∥MN∥) ∧ (∥D - H∥ / ∥MN∥ < sqrt 2 / 4) :=
sorry

end problem1_problem2_l239_239438


namespace area_difference_inside_circle_outside_triangle_l239_239782

theorem area_difference_inside_circle_outside_triangle
  (r : ℝ) (s : ℝ) (A : ℝ) (h_r : r = 3) (h_s : s = 6) (h_A : A = 0) :
  let circle_area : ℝ := π * r^2
  let triangle_area : ℝ := (√3 / 4) * s^2
  circle_area - triangle_area = 9 * (π - √3) :=
by sorry

end area_difference_inside_circle_outside_triangle_l239_239782


namespace barbara_shopping_l239_239350

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l239_239350


namespace largest_lcm_l239_239274

theorem largest_lcm :
  let l4 := Nat.lcm 18 3
  let l5 := Nat.lcm 18 6
  let l6 := Nat.lcm 18 9
  let l7 := Nat.lcm 18 12
  let l8 := Nat.lcm 18 15
  let l9 := Nat.lcm 18 18
  in max (max (max (max (max l4 l5) l6) l7) l8) l9 = 90 := by
    sorry

end largest_lcm_l239_239274


namespace modulus_of_z_l239_239585

noncomputable def z : ℂ :=
  (1 + 7 * complex.I) / (2 - complex.I)

theorem modulus_of_z :
  |z| = real.sqrt 10 := sorry

end modulus_of_z_l239_239585


namespace negative_number_among_options_l239_239335

theorem negative_number_among_options :
  ∀ (x : ℝ), x ∈ {abs (-3), -(-3), (-3)^2, -real.sqrt 3} → x = -real.sqrt 3 :=
by sorry

end negative_number_among_options_l239_239335


namespace distribute_tickets_among_people_l239_239377

noncomputable def distribution_ways : ℕ := 84

theorem distribute_tickets_among_people (tickets : Fin 5 → ℕ) (persons : Fin 4 → ℕ)
  (h1 : ∀ p : Fin 4, ∃ t : Fin 5, tickets t = persons p)
  (h2 : ∀ p : Fin 4, ∀ t1 t2 : Fin 5, tickets t1 = persons p ∧ tickets t2 = persons p → (t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val)) :
  ∃ n : ℕ, n = distribution_ways := by
  use 84
  trivial

end distribute_tickets_among_people_l239_239377


namespace circumcircle_tangent_to_BC_l239_239167

variable {A B C P M N R S : Type}
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited M] [Inhabited N] [Inhabited R] [Inhabited S]

-- Definitions and conditions given in the problem
variables (A B C P : Point)
variables (M : Point) (hM : M ∈ Segment A B)
variables (N : Point) (hN : N ∈ Segment A C)
variables (hP : P ∈ Line B C)
variables (hMN_not_parallel : ¬Parallel (Line M N) (Line B C))
variables (hParallelogram : Parallelogram A M P N)
variables (R S : Point) (hRS : R ≠ S ∧ R ∈ Circle A B C ∧ S ∈ Circle A B C ∧ Collinear [M, N, R] ∧ Collinear [M, N, S])

-- Question to prove
theorem circumcircle_tangent_to_BC :
  Tangent (Circumcircle R P S) (Line B C) := sorry

end circumcircle_tangent_to_BC_l239_239167


namespace last_digit_of_4_over_3_power_5_l239_239275

noncomputable def last_digit_of_fraction (n d : ℕ) : ℕ :=
  (n * 10^5 / d) % 10

def four : ℕ := 4
def three_power_five : ℕ := 3^5

theorem last_digit_of_4_over_3_power_5 :
  last_digit_of_fraction four three_power_five = 7 :=
by
  sorry

end last_digit_of_4_over_3_power_5_l239_239275


namespace probability_all_four_stop_same_toss_l239_239600

noncomputable def prob_two_consecutive_heads (n : ℕ) := (1 / 2 ^ n)

theorem probability_all_four_stop_same_toss :
  ∑' n : ℕ, if h : n ≥ 2 then (prob_two_consecutive_heads n) ^ 4 else 0 = 1 / 240 :=
by
  sorry

end probability_all_four_stop_same_toss_l239_239600


namespace find_x_l239_239100

-- Definitions based on conditions
variables (A B C M O : Type)
variables (OA OB OC OM : vector_space O)
variables (x : ℚ) -- Rational number type for x

-- Condition (1): M lies in the plane ABC
-- Condition (2): OM = x * OA + 1/3 * OB + 1/2 * OC
axiom H : OM = x • OA + (1 / 3 : ℚ) • OB + (1 / 2 : ℚ) • OC

-- The theorem statement
theorem find_x :
  x = 1 / 6 :=
sorry -- Proof is to be provided

end find_x_l239_239100


namespace part1_inequality_part2_range_of_a_l239_239531

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239531


namespace find_vector_at_t5_l239_239793

def vector_on_line (t : ℝ) : ℝ × ℝ := 
  let a := (0, 11) -- From solving the system of equations
  let d := (2, -4) -- From solving the system of equations
  (a.1 + t * d.1, a.2 + t * d.2)

theorem find_vector_at_t5 : vector_on_line 5 = (10, -9) := 
by 
  sorry

end find_vector_at_t5_l239_239793


namespace largest_number_among_set_largest_number_among_set_l239_239337

theorem largest_number_among_set : 
  ∀ (S : set ℤ), S = {0, -1, -2, 1} → ∃ x ∈ S, ∀ y ∈ S, x ≥ y := 
by 
  intro S hS,
  have h : 1 ∈ S := by rw [hS]; simp,
  use 1,
  split,
  { exact h },
  { intro y hy,
    rw [hS] at hy,
    fin_cases hy;
    norm_num }

-- Or alternatively, a more streamlined and broader approach:
theorem largest_number_among_set :
  ∃ x ∈ ({0, -1, -2, 1} : set ℤ), ∀ y ∈ ({0, -1, -2, 1} : set ℤ), x ≥ y :=
by
  use 1
  split
  { simp }
  { intros y hy
    fin_cases hy
    all_goals { norm_num } }

end largest_number_among_set_largest_number_among_set_l239_239337


namespace variance_scaled_data_l239_239589

open Real

variable {a : ℕ → ℝ} {n : ℕ}

noncomputable def variance (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / n) * (finset.sum (finset.range n) (λ i, (a i - finset.sum (finset.range n) a / n) ^ 2))

theorem variance_scaled_data (h : variance a 6 = 2) :
  variance (λ i, 2 * a i) 6 = 8 :=
sorry

end variance_scaled_data_l239_239589


namespace assign_workers_l239_239788

theorem assign_workers (workers : Fin 5) (positions : Fin 3)
  (assign : workers → positions) :
  (∀ p : positions, 1 ≤ (Fintype.card {w // assign w = p})) → 
  (∃ f : {A : workers} × {B : workers} → by sorry) -- Place holder for the actual existence quantifier that makes sense only when we deal with the actual proof logic.
  := sorry

end assign_workers_l239_239788


namespace sin_double_angle_neg_l239_239948

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239948


namespace original_proposition_false_converse_false_inverse_false_contrapositive_false_l239_239758

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop := 
  (a * b ≤ 0) → (a ≤ 0 ∨ b ≤ 0)

-- Define the converse
def converse (a b : ℝ) : Prop := 
  (a ≤ 0 ∨ b ≤ 0) → (a * b ≤ 0)

-- Define the inverse
def inverse (a b : ℝ) : Prop := 
  (a * b > 0) → (a > 0 ∧ b > 0)

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := 
  (a > 0 ∧ b > 0) → (a * b > 0)

-- Prove that the original proposition is false
theorem original_proposition_false : ∀ (a b : ℝ), ¬ original_proposition a b :=
by sorry

-- Prove that the converse is false
theorem converse_false : ∀ (a b : ℝ), ¬ converse a b :=
by sorry

-- Prove that the inverse is false
theorem inverse_false : ∀ (a b : ℝ), ¬ inverse a b :=
by sorry

-- Prove that the contrapositive is false
theorem contrapositive_false : ∀ (a b : ℝ), ¬ contrapositive a b :=
by sorry

end original_proposition_false_converse_false_inverse_false_contrapositive_false_l239_239758


namespace largest_lcm_l239_239267

theorem largest_lcm :
  let lcm_value := List.map (λ b => Nat.lcm 18 b) [3, 6, 9, 12, 15, 18]
  in List.maximum! lcm_value = 90 := by
  sorry

end largest_lcm_l239_239267


namespace part1_part2_l239_239543

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239543


namespace best_of_five_advantageous_l239_239327

theorem best_of_five_advantageous (p : ℝ) (h : p > 0.5) :
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    p2 > p1 :=
by 
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    sorry -- an actual proof would go here

end best_of_five_advantageous_l239_239327


namespace area_of_sector_l239_239234

-- Definitions based on conditions in the problem
def central_angle : ℝ := π / 6
def radius : ℝ := 2

-- Theorem statement
theorem area_of_sector (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle) :
  (1 / 2) * θ * r^2 = π / 3 := by
  sorry

end area_of_sector_l239_239234


namespace determine_lambda_l239_239112

def vector (α : Type _) [Add α] [Mul α] := α × α

variables {α : Type _} [Real α] [NormedAddCommGroup (vector α)] [InnerProductSpace α (vector α)]

noncomputable def vector_add (u v : vector α) : vector α := (u.1 + v.1, u.2 + v.2)
noncomputable def vector_sub (u v : vector α) : vector α := (u.1 - v.1, u.2 - v.2)
noncomputable def vector_dot (u v : vector α) : α := u.1 * v.1 + u.2 * v.2

variable (λ : α)
def a : vector α := (λ, 1)
def b : vector α := (λ + 2, 1)

theorem determine_lambda (h : ∥vector_add a b∥ = ∥vector_sub a b∥) : λ = -1 :=
sorry

end determine_lambda_l239_239112


namespace max_popsicles_l239_239645

theorem max_popsicles (total_money : ℕ) (popsicle_cost : ℕ) (discount_threshold : ℕ) (discount : ℕ) (total_money = 4575) (popsicle_cost = 225) (discount_threshold = 10) (discount = 5) :
  let discounted_cost := popsicle_cost - discount in
  let full_price_popsicles := total_money / popsicle_cost in
  let max_discounted_popsicles := (total_money - discount_threshold * popsicle_cost) / discounted_cost in
  max full_price_popsicles (discount_threshold + max_discounted_popsicles) = 20 :=
sorry

end max_popsicles_l239_239645


namespace cooler1_water_left_l239_239757

noncomputable def waterLeftInFirstCooler (gallons1 gallons2 : ℝ) (chairs rows : ℕ) (ozSmall ozLarge ozPerGallon : ℝ) : ℝ :=
  let totalChairs := chairs * rows
  let totalSmallOunces := totalChairs * ozSmall
  let initialOunces1 := gallons1 * ozPerGallon
  initialOunces1 - totalSmallOunces

theorem cooler1_water_left :
  waterLeftInFirstCooler 4.5 3.25 12 7 4 8 128 = 240 :=
by
  sorry

end cooler1_water_left_l239_239757


namespace solution_set_of_inequality_l239_239440

def even_function (f: ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def monotonically_increasing (f: ℝ → ℝ) (I: Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≤ f y

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : monotonically_increasing f {x | 0 < x})
  (h3 : f 1 = 0) :
  {x | x * f x < 0} = { x | x ∈ (Set.Ioo 0 1) ∪ (Set.Iio (-1)) } :=
begin
  sorry
end

end solution_set_of_inequality_l239_239440


namespace imaginary_part_of_z_l239_239695

def complex_imaginary_part : Type := ℂ

def z : complex_imaginary_part := (1 + complex.I) ^ 2 * (2 + complex.I)

theorem imaginary_part_of_z : z.im = 4 := by
  -- proof would go here
  sorry

end imaginary_part_of_z_l239_239695


namespace sin_double_angle_fourth_quadrant_l239_239970

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239970


namespace hair_cut_second_day_l239_239849

variable (hair_first_day : ℝ) (total_hair_cut : ℝ)

theorem hair_cut_second_day (h1 : hair_first_day = 0.375) (h2 : total_hair_cut = 0.875) :
  total_hair_cut - hair_first_day = 0.500 :=
by sorry

end hair_cut_second_day_l239_239849


namespace probability_at_least_one_l239_239380

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end probability_at_least_one_l239_239380


namespace sum_S_1_to_10_l239_239872

-- Define S_p for a given p
def S (p : ℕ) : ℕ := 15 * (60 * p + 87)

-- Define the sum of S_p for p from 1 to 10
def sum_S (n : ℕ) : ℕ := ∑ p in Finset.range n + 1, S p

theorem sum_S_1_to_10 : sum_S 10 = 62550 :=
by 
  -- Expected output
  sorry

end sum_S_1_to_10_l239_239872


namespace converse_and_inverse_false_l239_239927

variable (Polygon : Type)
variable (RegularHexagon : Polygon → Prop)
variable (AllSidesEqual : Polygon → Prop)

theorem converse_and_inverse_false (p : Polygon → Prop) (q : Polygon → Prop)
  (h : ∀ x, RegularHexagon x → AllSidesEqual x) :
  ¬ (∀ x, AllSidesEqual x → RegularHexagon x) ∧ ¬ (∀ x, ¬ RegularHexagon x → ¬ AllSidesEqual x) :=
by
  sorry

end converse_and_inverse_false_l239_239927


namespace sum_2n_plus_1_eq_n_plus_1_times_2n_plus_1_l239_239214

theorem sum_2n_plus_1_eq_n_plus_1_times_2n_plus_1 (n : ℕ) : 
  (finset.range (2 * n + 2)).sum (λ x, x + 1) = (n + 1) * (2 * n + 1) :=
begin
  induction n with k hk,
  {
    -- Base case: n = 0
    rw [finset.sum_range_succ, finset.sum_range_zero], 
    simp,
  },
  {
    -- Induction step: Assume the statement holds for some k, show it holds for k + 1
    have hsum : (finset.range (2 * k + 2)).sum (λ x, x + 1) = (k + 1) * (2 * k + 1), from hk,
    rw finset.sum_range_succ,
    rw finset.sum_range_succ,
    rw hsum, 
    simp,
  },
  sorry
end

end sum_2n_plus_1_eq_n_plus_1_times_2n_plus_1_l239_239214


namespace part1_solution_set_part2_range_of_a_l239_239534

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239534


namespace quadratic_equation_root_l239_239385

def has_rational_coefficients (a b c : ℚ) := 
  ∃ p q r : ℤ, a = p ∧ b = q ∧ c = r

theorem quadratic_equation_root (a b c : ℚ) (h_rational : has_rational_coefficients a b c)
  (h_quad : a = 1) (h_root : Polynomial.eval (Real.sqrt 5 - 3) (Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c) = 0) :
  a = 1 ∧ b = 6 ∧ c = -4 :=
by
  sorry

end quadratic_equation_root_l239_239385


namespace min_value_of_2x_plus_y_l239_239124

theorem min_value_of_2x_plus_y (x y : ℝ) (h : log 2 x + log 2 y = 3) : 2 * x + y ≥ 8 :=
sorry

end min_value_of_2x_plus_y_l239_239124


namespace hyperbola_focal_length_l239_239478

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239478


namespace proof_equiv_l239_239435

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 2^(x + 1) > 4}

def C : Set ℝ := {x | x ≤ 1}

theorem proof_equiv : A ∩ C = {x | -1 ≤ x ∧ x ≤ 1} := 
sorry

end proof_equiv_l239_239435


namespace imaginary_part_of_one_plus_i_five_l239_239633

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the problem statement: Proving that the imaginary part of (1+i)^5 is -4.
theorem imaginary_part_of_one_plus_i_five : Complex.I = i → Complex.im ((1 + i) ^ 5) = -4 :=
by
  -- implementation of the proof
  intro hi,
  sorry

end imaginary_part_of_one_plus_i_five_l239_239633


namespace isosceles_right_triangle_APQ_l239_239211

-- Definitions of the geometric objects and conditions
variables (A B C D E F G H K P Q : Type) [triangle ABC : triangle ℝ A B C]
variable (BCDE : square ℝ B C D E)
variable (ACFG : square ℝ A C F G)
variable (BAHK : square ℝ B A H K)
variable (FCDQ : parallelogram ℝ F C D Q)
variable (EBKP : parallelogram ℝ E B K P)

-- Main theorem statement
theorem isosceles_right_triangle_APQ (h1 : square. constructed outwardly_on_side A B H K)
                                     (h2 : square. constructed outwardly_on_side B C D E)
                                     (h3 : square. constructed outwardly_on_side A C F G)
                                     (h4 : parallelogram. constructed F C D Q)
                                     (h5 : parallelogram. constructed E B K P)
                                     : isosceles_right_triangle ℝ A P Q :=
sorry

end isosceles_right_triangle_APQ_l239_239211


namespace volume_of_circumscribed_sphere_l239_239992

theorem volume_of_circumscribed_sphere (S : ℝ) (V : ℝ) 
  (h1 : S = 6) : V = (3.sqrt / 2) * π := 
sorry

end volume_of_circumscribed_sphere_l239_239992


namespace product_of_all_possible_values_of_b_l239_239003

def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def f_inv (b : ℝ) (y : ℝ) : ℝ := sorry  -- Define inverse manually if needed

theorem product_of_all_possible_values_of_b :
  ∀ b : ℝ, (f(b) 3 = f_inv(b) (b + 2)) → b = -2 := 
by
  intros b h
  sorry  -- The proof is not required in the task

end product_of_all_possible_values_of_b_l239_239003


namespace cake_slices_l239_239811

open Nat

theorem cake_slices (S : ℕ) (h1 : 2 * S - 12 = 10) : S = 8 := by
  sorry

end cake_slices_l239_239811


namespace inequality_solution_set_l239_239664

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 4 - 2 * (2 * x ^ 2 + 7 * x + 3) ^ (1 / 2)) *
            (|x ^ 2 - 4 * x + 2| - |x - 2|) ≤ 0} =
  {x : ℝ | x ∈ Icc (-∞) (-3) ∪ Icc 0 1 ∪ {2} ∪ Icc 3 4} :=
by
  sorry

end inequality_solution_set_l239_239664


namespace max_value_of_a_plus_2b_l239_239564

variable (a b c : ℝ)

def condition := a^2 + 2 * b^2 + 3 * c^2 = 1

theorem max_value_of_a_plus_2b (h : condition a b c) : a + 2 * b ≤ √3 :=
sorry

end max_value_of_a_plus_2b_l239_239564


namespace find_k2_minus_b2_l239_239914

theorem find_k2_minus_b2 (k b : ℝ) (h1 : 3 = k * 1 + b) (h2 : 2 = k * (-1) + b) : k^2 - b^2 = -6 := 
by
  sorry

end find_k2_minus_b2_l239_239914


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239966

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239966


namespace find_BE_l239_239612

theorem find_BE (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  {AB BC CA : ℝ} (h1 : AB = 10) (h2 : BC = 12) (h3 : CA = 11)
  (h4 : segment_contains D B C) (h5 : dist C D = 5)
  (h6 : angle B A E = angle C A D) :
  dist B E = 40 / 7 := by sorry

end find_BE_l239_239612


namespace smallest_solution_l239_239062

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239062


namespace am_gm_inequality_for_x_l239_239639

theorem am_gm_inequality_for_x (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by 
  sorry

end am_gm_inequality_for_x_l239_239639


namespace minimum_point_coordinates_l239_239692

open Real

noncomputable def original_function (x : ℝ) : ℝ :=
  abs x ^ 2 - 3

noncomputable def translated_function (x : ℝ) : ℝ :=
  original_function (x - 1) - 4

theorem minimum_point_coordinates :
  (∃ x y : ℝ, translated_function x = y ∧ ∀ z : ℝ, translated_function z ≥ y ∧ (x, y) = (1, -7)) :=
by
  sorry

end minimum_point_coordinates_l239_239692


namespace problem_a_problem_b_l239_239765

-- Define necessary elements for the problem
def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the method to check divisibility by seven
noncomputable def check_divisibility_by_seven (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  remaining_digits - 2 * last_digit

-- Problem a: Prove that 4578 is divisible by 7
theorem problem_a : is_divisible_by_seven 4578 :=
  sorry

-- Problem b: Prove that there are 13 three-digit numbers of the form AB5 divisible by 7
theorem problem_b : ∃ (count : ℕ), count = 13 ∧ (∀ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 → is_divisible_by_seven (100 * a + 10 * b + 5) → count = count + 1) :=
  sorry

end problem_a_problem_b_l239_239765


namespace shortest_distance_from_point_to_segment_l239_239750

noncomputable def distance_from_point_to_segment (P A B : EuclideanSpace ℝ (Fin 2)) : ℝ :=
if (perpendicular_zone P A B) then 
    perpendicular_distance P A B
else 
    min (euclidean_distance P A) (euclidean_distance P B)

theorem shortest_distance_from_point_to_segment (P A B : EuclideanSpace ℝ (Fin 2)) : 
  (∃ d : ℝ, d = distance_from_point_to_segment P A B) :=
begin
    sorry
end

end shortest_distance_from_point_to_segment_l239_239750


namespace total_cost_other_goods_l239_239353

/-- The total cost of the goods other than tuna and water -/
theorem total_cost_other_goods :
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  cost_other = 40 :=
by
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  show cost_other = 40
  sorry

end total_cost_other_goods_l239_239353


namespace original_profit_percentage_l239_239322

theorem original_profit_percentage (C : ℝ) (C' : ℝ) (S' : ℝ) (H1 : C = 40) (H2 : C' = 32) (H3 : S' = 41.60) 
  (H4 : S' = (1.30 * C')) : (S' + 8.40 - C) / C * 100 = 25 := 
by 
  sorry

end original_profit_percentage_l239_239322


namespace angle_F_is_28_l239_239614

variables (F E D : ℝ)
variables (hD : D = 75) 
variables (hE : E = 4 * F - 37)
variables (hSum : D + E + F = 180)

theorem angle_F_is_28 : F = 28 :=
by 
  have h1 : 75 = D := hD
  have h2 : E = 4 * F - 37 := hE
  have h3 : D + E + F = 180 := hSum
  rw ←h1 at h3
  rw ←h2 at h3
  sorry  -- completing the proof is not required.

end angle_F_is_28_l239_239614


namespace part1_part2_l239_239888

noncomputable def z (m: ℝ) : ℂ := ⟨m^2 - m - 6, m^2 - 11 * m + 24⟩

theorem part1 (m : ℝ) (hz : z m = ⟨0, m^2 - 11 * m + 24⟩) : m = -2 :=
by {
  have h_real : m^2 - m - 6 = 0,
  { exact Complex.ext_iff.mp hz.left },
  sorry
}

theorem part2 (m : ℝ) (hz : z m ∈ {z : ℂ | z.re < 0 ∧ z.im > 0}) : -2 < m ∧ m < 3 :=
by {
  have h_real_le : m^2 - m - 6 < 0,
  { exact hz.left },
  have h_im_ge : m^2 - 11 * m + 24 > 0,
  { exact hz.right },
  sorry
}

end part1_part2_l239_239888


namespace cos_double_angle_l239_239101

def x : ℝ := -1
def y : ℝ := 2
def r : ℝ := Real.sqrt (x^2 + y^2)
def cos_alpha : ℝ := x / r
def sin_alpha : ℝ := y / r

theorem cos_double_angle : (cos_alpha^2 - sin_alpha^2) = -3 / 5 := by
  sorry

end cos_double_angle_l239_239101


namespace parallel_line_distance_l239_239913

theorem parallel_line_distance (m n : ℝ) 
    (h₀ : ∀ x y : ℝ, 3 * x - y + m = 0 → 6 * x + n * y + 7 = 0 ∧ distance = sqrt 10 / 4):
  (m = 6 ∨ m = 1) :=
sorry

end parallel_line_distance_l239_239913


namespace no_solution_eq_l239_239070

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l239_239070


namespace distinct_elements_in_T_l239_239638

open Set

variable {n k : ℕ}
variable (S : Set ℝ)

def isDistinct (s : ℝ → Prop) : Prop :=
  ∀ x y, s x → s y → x ≠ y → x ≠ y

def T (S : Set ℝ) (n k : ℕ) : Set ℝ :=
  {t | ∃ (x : ℕ → ℝ) (h : ∀ i : ℕ, i < k → S x i ∧ isDistinct (λ i, x i)), t = (Finset.sum (Finset.range k) (λ j, x j))}

theorem distinct_elements_in_T (n k : ℕ) (hk : k ≤ n) (S : Set ℝ)
  (hS : S.finite ∧ S.card = n ∧ isDistinct (λ x, x ∈ S)) :
  ∀ T : Set ℝ, (T = {t | ∃ x (h : ∀ i, i < k → S x i ∧ isDistinct (λ i, x i)), t = (Finset.sum (Finset.range k) (λ j, x j))}) →
  (T.countable) →
  T.card ≥ k * (n - k) + 1 :=
sorry

end distinct_elements_in_T_l239_239638


namespace function_domain_l239_239706

theorem function_domain (x : ℝ) : (y = x / (5-x)) → x ≠ 5 :=
by
  intros
  unfold
  sorry

end function_domain_l239_239706


namespace limit_transformed_l239_239102

noncomputable def limit_derivative (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  limit (λ (Δx : ℝ), (f (x0 + Δx) - f x0) / Δx) (0 : ℝ) (3 : ℝ)

theorem limit_transformed (f : ℝ → ℝ) (x0 : ℝ) (h : limit_derivative f x0) :
  limit (λ Δx : ℝ, (f (x0 + Δx) - f x0) / (3 * Δx)) (0 : ℝ) (1 : ℝ) :=
sorry

end limit_transformed_l239_239102


namespace Beethoven_birth_day_l239_239228

theorem Beethoven_birth_day :
  (exists y: ℤ, y.mod 4 = 0 ∧ y.mod 100 ≠ 0 ∨ y.mod 400 = 0 → y ∈ range(1770, 2021) → y)
  → (∃ d: Zmod 7, d = (16 : Zmod 7) - 2) :=
by
  let totalYears := 250
  let leapYears := 60
  let regularYears := 190
  let dayShifts := regularYears + 2 * leapYears
  have modDays : dayShifts % 7 = 2 := by

  -- Proof skipped
  sorry

end Beethoven_birth_day_l239_239228


namespace alison_lollipops_l239_239574

variable (A D H : ℕ)

theorem alison_lollipops : A = 60 :=
  have h1: H = A + 30 := sorry,
  have h2: A = D / 2 := sorry,
  have h3: A + H + D = 270 := sorry,

  calc
  A = 60 : sorry

end alison_lollipops_l239_239574


namespace a_10_eq_neg_p_S_2018_eq_p_plus_q_l239_239427

variables (p q : ℝ)
def sequence (n : ℕ) : ℝ :=
  match n with
  | 0     => 0 -- undefined term (sequence starts at 1)
  | 1     => p
  | 2     => q
  | (n+3) => sequence p q (n + 2) - sequence p q (n + 1)
  end

def sum_sequence (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sequence p q (i + 1)

theorem a_10_eq_neg_p : sequence p q 10 = -p := sorry

theorem S_2018_eq_p_plus_q : sum_sequence p q 2018 = p + q := sorry

end a_10_eq_neg_p_S_2018_eq_p_plus_q_l239_239427


namespace sin_double_angle_in_fourth_quadrant_l239_239940

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239940


namespace diagonals_of_square_equal_l239_239019

-- Definitions and conditions
variable (Rectangle Square : Type)
variable (isRectangle : Square → Rectangle)
variable (diagonalsEqual : ∀ (r : Rectangle), ∀ d1 d2, r.diagonals d1 = d2)

-- Proof statement
theorem diagonals_of_square_equal (s : Square) (d1 d2 : Diagonal) 
  : s.diagonals d1 = d2 :=
begin
  have r := isRectangle s,
  exact diagonalsEqual r d1 d2,
end

end diagonals_of_square_equal_l239_239019


namespace conference_fraction_married_men_l239_239823

theorem conference_fraction_married_men 
  (total_women : ℕ) 
  (single_probability : ℚ) 
  (h_single_prob : single_probability = 3/7) 
  (h_total_women : total_women = 7) : 
  (4 : ℚ) / (11 : ℚ) = 4 / 11 := 
by
  sorry

end conference_fraction_married_men_l239_239823


namespace solution_set_inequality_l239_239397

theorem solution_set_inequality (a : ℝ) :
  ∀ x : ℝ,
    (12 * x^2 - a * x > a^2) →
    ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
     (a = 0 ∧ x ≠ 0) ∨
     (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
by
  sorry

end solution_set_inequality_l239_239397


namespace angle_C_ne_5pi_over_6_l239_239615

-- Define the triangle ∆ABC
variables (A B C : ℝ)

-- Assume the conditions provided
axiom condition_1 : 3 * Real.sin A + 4 * Real.cos B = 6
axiom condition_2 : 3 * Real.cos A + 4 * Real.sin B = 1

-- State that the size of angle C cannot be 5π/6
theorem angle_C_ne_5pi_over_6 : C ≠ 5 * Real.pi / 6 :=
sorry

end angle_C_ne_5pi_over_6_l239_239615


namespace gcd_five_triang_num_l239_239402

theorem gcd_five_triang_num (n : ℕ) (hn_pos : n > 0) :
    let T_n := (n * (n + 1)) / 2 in
    Nat.gcd (5 * T_n) (n - 1) = 1 :=
by { 
    sorry 
}

end gcd_five_triang_num_l239_239402


namespace part1_inequality_part2_range_of_a_l239_239528

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239528


namespace janes_stick_shorter_than_sarahs_l239_239654

theorem janes_stick_shorter_than_sarahs :
  ∀ (pat_length jane_length pat_dirt sarah_factor : ℕ),
    pat_length = 30 →
    jane_length = 22 →
    pat_dirt = 7 →
    sarah_factor = 2 →
    (sarah_factor * (pat_length - pat_dirt)) - jane_length = 24 :=
by
  intros pat_length jane_length pat_dirt sarah_factor h1 h2 h3 h4
  -- sorry skips the proof
  sorry

end janes_stick_shorter_than_sarahs_l239_239654


namespace sales_second_month_l239_239791

theorem sales_second_month 
  (sale_1 : ℕ) (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ)
  (avg_sale : ℕ)
  (h1 : sale_1 = 5400)
  (h2 : sale_3 = 6300)
  (h3 : sale_4 = 7200)
  (h4 : sale_5 = 4500)
  (h5 : sale_6 = 1200)
  (h_avg : avg_sale = 5600) :
  sale_2 = 9000 := 
by sorry

end sales_second_month_l239_239791


namespace area_of_sandbox_is_correct_l239_239326

-- Define the length and width of the sandbox
def length_sandbox : ℕ := 312
def width_sandbox : ℕ := 146

-- Define the area calculation
def area_sandbox (length width : ℕ) : ℕ := length * width

-- The theorem stating that the area of the sandbox is 45552 cm²
theorem area_of_sandbox_is_correct : area_sandbox length_sandbox width_sandbox = 45552 := sorry

end area_of_sandbox_is_correct_l239_239326


namespace evaluate_expression_at_100_l239_239382

theorem evaluate_expression_at_100 :
  let x : ℝ := 100
  let numerator : ℝ := 3 * x^3 - 7 * x^2 + 4 * x - 9
  let denominator : ℝ := 2 * x - 0.5
  numerator / denominator ≈ 14684.73534 :=
by sorry

end evaluate_expression_at_100_l239_239382


namespace lean_math_problem_l239_239086

noncomputable theory
open Real

variable {f : ℝ → ℝ}

-- Hypotheses
def differentiable_on_ℝ (f : ℝ → ℝ) := ∀ x : ℝ, differentiable_at ℝ f x

def condition1 (f : ℝ → ℝ) := differentiable_on_ℝ f
def condition2 (f : ℝ → ℝ) := ∀ x : ℝ, deriv f x + f x < 0

-- Conclusion
theorem lean_math_problem
  (h1 : condition1 f)
  (h2 : condition2 f) :
  ∀ m : ℝ, (f (m - m^2)) / (exp (m^2 - m + 1)) > f 1 :=
sorry

end lean_math_problem_l239_239086


namespace lucas_siblings_product_is_35_l239_239596

-- Definitions based on the given conditions
def total_girls (lauren_sisters : ℕ) : ℕ := lauren_sisters + 1
def total_boys (lauren_brothers : ℕ) : ℕ := lauren_brothers + 1

-- Given conditions
def lauren_sisters : ℕ := 4
def lauren_brothers : ℕ := 7

-- Compute number of sisters (S) and brothers (B) Lucas has
def lucas_sisters : ℕ := total_girls lauren_sisters
def lucas_brothers : ℕ := lauren_brothers

theorem lucas_siblings_product_is_35 : 
  (lucas_sisters * lucas_brothers = 35) := by
  -- Asserting the correctness based on given family structure conditions
  sorry

end lucas_siblings_product_is_35_l239_239596


namespace piecewise_function_continuity_at_3_l239_239878

def f (b x : ℝ) : ℝ :=
  if x > 3 then x + b else 2 * x + 2

theorem piecewise_function_continuity_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f b x - f b 3) < ε) ↔ b = 5 := 
by
  sorry

end piecewise_function_continuity_at_3_l239_239878


namespace vector_magnitude_subtraction_l239_239569

variables (a b : ℝ^3)
variables (h1 : dot_product a b = 1)
variables (h2 : norm a = 1)
variables (h3 : norm b = 2)

theorem vector_magnitude_subtraction : norm (a - b) = real.sqrt 3 :=
  sorry

end vector_magnitude_subtraction_l239_239569


namespace graph_forms_l239_239841

theorem graph_forms (x y : ℝ) :
  x^3 * (2 * x + 2 * y + 3) = y^3 * (2 * x + 2 * y + 3) →
  (∀ x y : ℝ, y ≠ x → y = -x - 3 / 2) ∨ (y = x) :=
sorry

end graph_forms_l239_239841


namespace total_songs_l239_239288

-- Define the number of albums Faye bought and the number of songs per album
def country_albums : ℕ := 2
def pop_albums : ℕ := 3
def songs_per_album : ℕ := 6

-- Define the total number of albums Faye bought
def total_albums : ℕ := country_albums + pop_albums

-- Prove that the total number of songs Faye bought is 30
theorem total_songs : total_albums * songs_per_album = 30 := by
  sorry

end total_songs_l239_239288


namespace infinitely_many_n_divisible_by_n_squared_l239_239220

theorem infinitely_many_n_divisible_by_n_squared :
  ∃ (n : ℕ → ℕ), (∀ k : ℕ, 0 < n k) ∧ (∀ k : ℕ, n k^2 ∣ 2^(n k) + 3^(n k)) :=
sorry

end infinitely_many_n_divisible_by_n_squared_l239_239220


namespace james_bought_five_shirts_l239_239161

theorem james_bought_five_shirts (total_cost : ℝ) (discount : ℝ) (discounted_price : ℝ) (N : ℝ) 
    (h1 : total_cost = 60) (h2 : discount = 0.4) (h3 : discounted_price = 12) :
    N = 5 :=
by
  have original_price := discounted_price / (1 - discount)
  have num_shirts := total_cost / discounted_price
  have original_price_value : original_price = 20 := by sorry -- This line is not necessary for the main theorem
  have num_shirts_value : num_shirts = 5 := by sorry -- This line is not necessary for the main theorem
  exact num_shirts_value

end james_bought_five_shirts_l239_239161


namespace multiplier_of_product_l239_239581

variable {a b : ℝ}

theorem multiplier_of_product (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a + b = k * (a * b))
  (h4 : (1 / a) + (1 / b) = 6) : k = 6 := by
  sorry

end multiplier_of_product_l239_239581


namespace remaining_letter_orders_l239_239601

theorem remaining_letter_orders:
  let T := {1, 2, 3, 4, 6, 7, 8, 10}
  ∑ k in (finset.range 9), (nat.choose 8 k) * (k + 2) = 1400 := by
  sorry

end remaining_letter_orders_l239_239601


namespace smallest_solution_exists_l239_239048

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239048


namespace true_q_if_not_p_and_p_or_q_l239_239588

variables {p q : Prop}

theorem true_q_if_not_p_and_p_or_q (h1 : ¬p) (h2 : p ∨ q) : q :=
by 
  sorry

end true_q_if_not_p_and_p_or_q_l239_239588


namespace hyperbola_focal_length_is_4_l239_239444

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239444


namespace algebraic_expression_value_l239_239083

-- Define the conditions
variables (x y : ℝ)
-- Condition 1: x - y = 5
def cond1 : Prop := x - y = 5
-- Condition 2: xy = -3
def cond2 : Prop := x * y = -3

-- Define the statement to be proved
theorem algebraic_expression_value :
  cond1 x y → cond2 x y → x^2 * y - x * y^2 = -15 :=
by
  intros h1 h2
  sorry

end algebraic_expression_value_l239_239083


namespace biased_coin_flips_l239_239755

theorem biased_coin_flips (p : ℚ) (h : 0 < p ∧ p ≤ 1)
  (h_eq : 7 * p * (1 - p) ^ 6 = 21 * p^2 * (1 - p) ^ 5) :
  let prob_4_heads := 35 * (1 / 4)^4 * (3 / 4)^3,
      t := rat.mk_pnat 945 (pnat.of_nat 16384) in
  (prob_4_heads.denom + prob_4_heads.num) = 17329 := 
by
  sorry

end biased_coin_flips_l239_239755


namespace smallest_solution_l239_239044

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239044


namespace no_unique_sum_subset_l239_239169

-- Define M as the rationals in the interval (0, 1)
def M : set ℚ := { x : ℚ | 0 < x ∧ x < 1 }

-- Define the problem statement
theorem no_unique_sum_subset :
  ¬ ∃ A ⊆ M, ∀ x ∈ M, ∃! (S : finset ℚ), S ⊆ A ∧ S.sum id = x :=
by 
sorry

end no_unique_sum_subset_l239_239169


namespace winner_collected_l239_239850

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end winner_collected_l239_239850


namespace triangle_BC_range_l239_239098

open Real

variable {a C : ℝ} (A : ℝ) (ABC : Triangle A C)

/-- Proof problem statement -/
theorem triangle_BC_range (A C : ℝ) (h0 : 0 < A) (h1 : A < π) (c : ℝ) (h2 : c = sqrt 2) (h3 : a * cos C = c * sin A): 
  ∃ (BC : ℝ), sqrt 2 < BC ∧ BC < 2 :=
sorry

end triangle_BC_range_l239_239098


namespace line_and_hyperbola_intersect_l239_239794

theorem line_and_hyperbola_intersect 
  (a b m : ℝ) 
  (line_eq : ∀ x y, y = x + m) 
  (e : ℝ := √3)
  (h_eccentricity : e = √3)
  (b2_eq_2a2 : b^2 = 2 * a^2)
  (line_slope : ∀ x y, y = x + 1)
  (P Q R : Point)
  (R_on_y_axis : R.coordinates.y = 0)
  (OP_dot_OQ : ((P.coordinates.x) * (Q.coordinates.x) + (P.coordinates.y) * (Q.coordinates.y)) = -3)
  (PR_eq_3RQ : P.coordinates.x / (Q.coordinates.x + 3) = 3) : 
  (∃ (l_eq : ∀ x y, y = x + 1) (h_eq : ∀ x y, 2 * x^2 - y^2 = 2), 
    (l_eq : ∀ x y, y = x + ±1) ∧ (∃ a: ℝ , ∀ x y, x^2 - y^2 / 2 = a)) := 
sorry

end line_and_hyperbola_intersect_l239_239794


namespace triangle_angle_sum_l239_239592

theorem triangle_angle_sum (A B C : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = π)
  (sin_ratio_condition : (sin A) / 2 = (sin B) / (sqrt 6) ∧ (sin B) / (sqrt 6) = (sin C) / (sqrt 3 + 1)) : 
  A + C = 2 * π / 3 :=
by
  sorry

end triangle_angle_sum_l239_239592


namespace quadratic_rational_coeff_l239_239387

theorem quadratic_rational_coeff (x : ℤ) : 
  (∃ (a b c : ℚ), a = 1 ∧ b = 6 ∧ c = 14 ∧ 
  (sqrt 5 - 3) ∈ {r | a * r^2 + b * r + c = 0}) :=
by
  use [1, 6, 14]
  split; norm_num
  sorry

end quadratic_rational_coeff_l239_239387


namespace focal_length_of_hyperbola_l239_239471

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239471


namespace part1_part2_l239_239546

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239546


namespace tank_capacity_l239_239764

-- Definitions and conditions from the problem
def leak_rate (C : ℝ) : ℝ := C / 7
def inlet_rate : ℝ := 360
def net_emptying_rate (C : ℝ) : ℝ := C / 12

-- The theorem corresponding to the problem statement
theorem tank_capacity : ∃ (C : ℝ), (leak_rate C = C / 7) ∧ (inlet_rate = 360) ∧ (net_emptying_rate C = C / 12) ∧ 
  360 - C / 7 = C / 12 ∧ 
  C = 1592 :=
by
  sorry

end tank_capacity_l239_239764


namespace distance_B_calculation_l239_239651

noncomputable def distance_between_O_and_B 
  (A O B : ℕ) 
  (d_AO : ℕ) 
  (t1 t2 : ℕ) 
  (v_Jia v_Yi : ℕ) 
  (equidistant_10_min : bool)
  (meet_40_min : bool) : ℕ :=
if equidistant_10_min ∧ meet_40_min then
  d_AO * v_Yi / v_Jia
else
  0

theorem distance_B_calculation (A O B : ℕ) 
  (d_AO : 1360 = (O - A : ℕ)) 
  (t1 t2 : 10 = t1 ∧ 40 = t2) 
  (v_Jia v_Yi : 34 = v_Jia ∧ 102 = v_Yi) 
  (equidistant_10_min : true)
  (meet_40_min : true) 
  : distance_between_O_and_B A O B d_AO t1 t2 v_Jia v_Yi equidistant_10_min meet_40_min = 2040 := 
sorry

end distance_B_calculation_l239_239651


namespace distance_from_point_to_line_l239_239499

variable (P A : ℝ × ℝ × ℝ)
variable (n : ℝ × ℝ × ℝ)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_length (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def point_to_line_distance (P A n : ℝ × ℝ × ℝ) : ℝ :=
  let AP := vector_sub P A
  let u := (n.1 / vector_length n, n.2 / vector_length n, n.3 / vector_length n)
  let proj_len := dot_product AP u
  let AP_len := vector_length AP
  Real.sqrt (AP_len^2 - proj_len^2)

theorem distance_from_point_to_line :
  P = (3, 3, 2) → A = (3, 2, 1) → n = (1, 0, 1) → point_to_line_distance P A n = Real.sqrt 6 / 2 :=
by
  intros hP hA hn
  subst hP
  subst hA
  subst hn
  sorry

end distance_from_point_to_line_l239_239499


namespace evaluate_expression_l239_239855

theorem evaluate_expression : -1 ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 = -2 :=
by
  -- sorry is added as a placeholder for the proof steps
  sorry

end evaluate_expression_l239_239855


namespace abs_val_problem_l239_239714

-- Define the problem statement
def abs_val_test : Prop :=
  |-1| = 1 

-- Prove that the expression evaluates to the given answer
theorem abs_val_problem : -|-1| = -1 :=
  by
    have abs_val_test : |-1| = 1,
    sorry

#check abs_val_problem

end abs_val_problem_l239_239714


namespace orange_probability_l239_239779

theorem orange_probability (total_apples : ℕ) (total_oranges : ℕ) (other_fruits : ℕ)
  (h1 : total_apples = 20) (h2 : total_oranges = 10) (h3 : other_fruits = 0) :
  (total_oranges : ℚ) / (total_apples + total_oranges + other_fruits) = 1 / 3 :=
by
  sorry

end orange_probability_l239_239779


namespace tangent_line_eq_extreme_values_interval_l239_239519

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l239_239519


namespace proof_method_of_right_triangle_is_contradiction_l239_239249

theorem proof_method_of_right_triangle_is_contradiction (A B C : ℝ)
    (h1 : ∠C = 90)
    (sum_angles : ∠A + ∠B + ∠C = 180) :
    "Proof by Contradiction" = "Proof by Contradiction" := by
  sorry

end proof_method_of_right_triangle_is_contradiction_l239_239249


namespace hyperbola_focal_length_l239_239494

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239494


namespace second_alloy_amount_l239_239604

theorem second_alloy_amount (x : ℝ) :
  let chromium_first_alloy := 0.12 * 15
  let chromium_second_alloy := 0.08 * x
  let total_weight := 15 + x
  let chromium_percentage_new_alloy := (0.12 * 15 + 0.08 * x) / (15 + x)
  chromium_percentage_new_alloy = (28 / 300) →
  x = 30 := sorry

end second_alloy_amount_l239_239604


namespace rectangle_diagonal_l239_239276

theorem rectangle_diagonal (l w : ℝ) (hl : l = 40) (hw : w = 40 * Real.sqrt 2) :
  Real.sqrt (l^2 + w^2) = 40 * Real.sqrt 3 :=
by
  rw [hl, hw]
  sorry

end rectangle_diagonal_l239_239276


namespace tangent_line_equation_at_1_l239_239920

-- Define the function f and the point of tangency
def f (x : ℝ) : ℝ := x^2 + 2 * x
def p : ℝ × ℝ := (1, f 1)

-- Statement of the theorem
theorem tangent_line_equation_at_1 :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = f x → y - p.2 = a * (x - p.1)) ∧
               4 * (p.1 : ℝ) - (p.2 : ℝ) - 1 = 0 :=
by
  -- Skipping the proof
  sorry

end tangent_line_equation_at_1_l239_239920


namespace smallest_solution_l239_239054

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239054


namespace smallest_solution_to_equation_l239_239033

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239033


namespace sum_of_distinct_prime_factors_315_l239_239753

theorem sum_of_distinct_prime_factors_315 : 
  ∃ factors : List ℕ, factors = [3, 5, 7] ∧ 315 = 3 * 3 * 5 * 7 ∧ factors.sum = 15 :=
by
  sorry

end sum_of_distinct_prime_factors_315_l239_239753


namespace largest_lcm_l239_239268

theorem largest_lcm :
  let lcm_value := List.map (λ b => Nat.lcm 18 b) [3, 6, 9, 12, 15, 18]
  in List.maximum! lcm_value = 90 := by
  sorry

end largest_lcm_l239_239268


namespace carlos_coupon_usage_l239_239361

theorem carlos_coupon_usage (start_day : ℕ) (days_in_week : ℕ := 7) (interval : ℕ := 13)
                            (num_coupons : ℕ := 9) (closed_day : ℕ := 2) : 
  start_day = 6 → ∀ (n : ℕ), n < num_coupons →
  (start_day + n * interval) % days_in_week ≠ closed_day :=
by
  intros start_day_eq n n_lt_coupons
  rw start_day_eq
  sorry

end carlos_coupon_usage_l239_239361


namespace max_vertices_no_rectangle_l239_239738

/-- In a regular 2016-sided polygon (2016-gon), prove that the maximum number of vertices 
    that can be marked such that no four marked vertices form the vertices of a rectangle is 1009. -/
theorem max_vertices_no_rectangle (n : ℕ) (h : n = 2016) : 
  ∃ (m : ℕ), m = 1009 ∧ 
    ∀ (marked : finset (fin n)), 
      marked.card ≤ m → 
      (¬ ∃ (a b c d : fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
        (is_rectangle a b c d)) := 
begin 
  sorry 
end

/-- Predicate to determine if four vertices form a rectangle in a regular n-gon. -/
def is_rectangle (a b c d : fin 2016) : Prop := 
  ∃ (k : ℕ), k ∈ finset.range 1008 ∧ 
    ((a = fin.of_nat k) ∧ (b = fin.of_nat (k + 1008)) ∧ 
     (c = fin.of_nat (k + 1008 + 1)) ∧ (d = fin.of_nat (k + 1)) ∨ 
     (a = fin.of_nat (k + 1008)) ∧ (b = fin.of_nat k) ∧ 
     (c = fin.of_nat (k + 1)) ∧ (d = fin.of_nat (k + 1008 + 1)))

end max_vertices_no_rectangle_l239_239738


namespace complex_div_result_l239_239870

theorem complex_div_result : (4 - 2 * Complex.i) / ((1 + Complex.i)^2) = -1 - 2 * Complex.i := by
  have h1 : (1 + Complex.i)^2 = 2 * Complex.i := by
    -- The calculation for (1 + i)^2
    calc (1 + Complex.i)^2
      = 1^2 + 2*1*Complex.i + (Complex.i)^2 : by ring
      = 1 + 2*Complex.i + (-1) : by rw [Complex.i_sq]
      = 2 * Complex.i : by ring
  -- Use this simplification in the main goal
  rw [h1]
  -- Continue the proof (not required in this statement)
  sorry

end complex_div_result_l239_239870


namespace sum_of_first_two_digits_of_repeating_decimal_l239_239707

theorem sum_of_first_two_digits_of_repeating_decimal (c d : ℕ) (h : (c, d) = (3, 5)) : c + d = 8 :=
by 
  sorry

end sum_of_first_two_digits_of_repeating_decimal_l239_239707


namespace chameleons_all_white_l239_239652

theorem chameleons_all_white :
  ∀ (a b c : ℕ), a = 800 → b = 1000 → c = 1220 → 
  (a + b + c = 3020) → (a % 3 = 2) → (b % 3 = 1) → (c % 3 = 2) →
    ∃ k : ℕ, (k = 3020 ∧ (k % 3 = 1)) ∧ 
    (if k = b then a = 0 ∧ c = 0 else false) :=
by
  sorry

end chameleons_all_white_l239_239652


namespace find_FC_l239_239407

theorem find_FC
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 9)
  (h2 : CB = 9)
  (h3 : AB = 1 / 3 * AD)
  (h4 : ED = 2 / 3 * AD)
  (h5 : ∀ (x y z : Triangle), x ∼ y ∧ y ∼ z ∧ z ∼ x → True) -- Sim⋂ilarity of triangles
  : FC = 12 :=
sorry

end find_FC_l239_239407


namespace part1_part2_l239_239547

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239547


namespace travel_time_optimization_l239_239358

variable (D : ℝ) (v_w : ℝ) (v_b : ℝ)

theorem travel_time_optimization :
  (D / (2 * v_w)) = Real.max (D / (2 * v_w) + D / (2 * v_b)) (D / (2 * v_b) + D / (2 * v_w)) :=
sorry

end travel_time_optimization_l239_239358


namespace average_age_combined_l239_239678

theorem average_age_combined (n_A n_B : ℕ) (age_avg_A age_avg_B : ℕ)
  (h1 : n_A = 8) (h2 : age_avg_A = 35) (h3 : n_B = 6) (h4 : age_avg_B = 30) :
  let n := n_A + n_B in
  let total_age_A := n_A * age_avg_A in
  let total_age_B := n_B * age_avg_B in
  let total_age := total_age_A + total_age_B in
  let age_avg := total_age / n in
  age_avg = 32.86 := 
by
  sorry

end average_age_combined_l239_239678


namespace equation_of_C_OA_perp_OB_and_AB_length_l239_239607

-- Proof problem for Q1
theorem equation_of_C :
  ∀ P : (ℝ × ℝ), 
  dist P (0, -real.sqrt 3) + dist P (0, real.sqrt 3) = 4 → (P.1^2 + (P.2^2) / 4 = 1) := 
  sorry

-- Proof problem for Q2
theorem OA_perp_OB_and_AB_length :
  ∀ A B : (ℝ × ℝ), 
  ∀ k : ℝ, 
  A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ 
  A ∈ {P | P.1^2 + (P.2^2) / 4 = 1} ∧ 
  B ∈ {P | P.1^2 + (P.2^2) / 4 = 1} →
  (k = 1/2 ∨ k = -1/2) → 
  ∃ k = 1/2 ∨ k = -1/2, 
  (A.1 * B.1 + A.2 * B.2 = 0) ∧ 
  (dist A B = 4 * real.sqrt 65 / 17) := 
  sorry

end equation_of_C_OA_perp_OB_and_AB_length_l239_239607


namespace insurance_cost_ratio_l239_239013

def monthlyEarnings : ℝ := 6000
def houseRental : ℝ := 640
def foodExpense : ℝ := 380
def electricWaterBill : ℝ := monthlyEarnings / 4
def remaining : ℝ := 2280

theorem insurance_cost_ratio :
  ∃ (insuranceCost : ℝ), 
  (monthlyEarnings - (houseRental + foodExpense + electricWaterBill + insuranceCost) = remaining) ∧
  (insuranceCost / monthlyEarnings = 1 / 5) :=
sorry

end insurance_cost_ratio_l239_239013


namespace range_of_x_l239_239705

def valid_domain (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x ≠ 4)

theorem range_of_x : ∀ x : ℝ, valid_domain x ↔ (x ≤ 3) :=
by sorry

end range_of_x_l239_239705


namespace cone_height_l239_239694

theorem cone_height (d l : ℝ) (r h : ℝ) 
  (h_diam : d = 8)
  (h_slant : l = 5)
  (h_radius : r = d / 2) 
  (h_relation : h^2 + r^2 = l^2) : 
  h = 3 := 
by {
  have h_r : r = 4 := by simp [h_diam, h_radius],
  have h_l : l = 5 := by simp [h_slant],
  have h_eq : h^2 + 4^2 = 25 := by simp [h_relation, h_r, h_l],
  have h_sq : h^2 = 9 := by linarith [h_eq],
  have h_val : h = 3 := by rw [← sq_eq_sq h_sq, real.sqrt_sq, ← h_eq],
  simp [h_val]
  sorry
}

end cone_height_l239_239694


namespace distinct_matches_possible_l239_239716

open Nat

noncomputable def matches_needed (n m d : ℕ) : ℕ :=
  ceil ((dn.toRat - m.toRat) / (2d - 1).toRat)

theorem distinct_matches_possible (n m d : ℕ) (hn : 0 < n) (hm : 0 < m) (hd : 0 < d) :
  n.contestants → m.matches →
  (∀ c, d ≤ (participates_in c matches.count)) →
  exists k, k ≥ matches_needed n m d ∧ (∀ i j, i ≠ j → distinct_in k) :=
begin
  sorry
end

end distinct_matches_possible_l239_239716


namespace exists_integers_abcd_l239_239224

theorem exists_integers_abcd (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end exists_integers_abcd_l239_239224


namespace square_in_acute_triangle_l239_239092

theorem square_in_acute_triangle (P Q R : Point) (h1 : acute_triangle P Q R) :
  ∃ A B C D : Point, 
    A ∈ line_segment P Q ∧ 
    B ∈ line_segment Q R ∧ 
    C ∈ line_segment R P ∧ 
    D ∈ line_segment R P ∧ 
    square A B C D :=
sorry

end square_in_acute_triangle_l239_239092


namespace min_value_special_grid_cells_l239_239895

theorem min_value_special_grid_cells (n : ℕ) (hn : n ≥ 2) (grid : Type) [has_red_cells : RedCells grid] :
  ∃ N : ℕ, N = 1 + nat.ceil((n + 1) / 5) :=
begin
  sorry
end

end min_value_special_grid_cells_l239_239895


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239962

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239962


namespace ratio_of_areas_l239_239428

-- Define the given problem in lean
variables {a c : ℝ} (h : ℝ) (h1 h2 : ℝ)
variable (d : ℝ := sqrt (a * c))

-- height ratios from similar triangles
def h1_def : ℝ := h * (a - sqrt (a * c)) / (a - c)
def h2_def : ℝ := h * (sqrt (a * c) - c) / (a - c)

-- area calculations for top and bottom trapezoids
def Area_top : ℝ := 1 / 2 * (a + sqrt (a * c)) * h * ((a - sqrt (a * c)) / (a - c))
def Area_bottom : ℝ := 1 / 2 * (c + sqrt (a * c)) * h * ((sqrt (a * c) - c) / (a - c))

-- statement to prove the ratio of areas 
theorem ratio_of_areas (h : a ≠ c) : (Area_top h) / (Area_bottom h) = a / c :=
sorry

end ratio_of_areas_l239_239428


namespace sum_of_two_draws_with_replacement_l239_239717

theorem sum_of_two_draws_with_replacement :
  let balls := {1, 2, 3, 4, 5}
  let draws := (balls × balls)
  let sums := draws.map (λ (x, y) => x + y)
  let unique_sums := {sum | sum ∈ sums}
  unique_sums.card = 9 := by
  sorry

end sum_of_two_draws_with_replacement_l239_239717


namespace triangle_area_incicle_trisects_median_l239_239257

theorem triangle_area_incicle_trisects_median 
  (PQ QR RP : ℝ) 
  (H1 : PQ = 28) 
  (PS trisected : ℝ) 
  (area : ℝ) 
  (H2 : area = (196* sqrt 3)) 
  (H3 : prime (nat.prime 3)) 
  : 196 + 3 = 199 := 
by
  -- proof to be written
  sorry

end triangle_area_incicle_trisects_median_l239_239257


namespace largest_lcm_l239_239266

theorem largest_lcm :
  let lcm_value := List.map (λ b => Nat.lcm 18 b) [3, 6, 9, 12, 15, 18]
  in List.maximum! lcm_value = 90 := by
  sorry

end largest_lcm_l239_239266


namespace find_p_l239_239442

-- Lean 4 definitions corresponding to the conditions
variables {p a b x0 y0 : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0) (hx0 : x0 ≠ 0)
variables (hA : (y0^2 = 2 * p * x0) ∧ ((x0 / a)^2 - (y0 / b)^2 = 1))
variables (h_dist : x0 + x0 = p^2)
variables (h_ecc : (5^.half) = sqrt 5)

-- The proof problem
theorem find_p :
  p = 1 :=
by
  sorry

end find_p_l239_239442


namespace sin_double_angle_fourth_quadrant_l239_239953

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239953


namespace total_cost_other_goods_l239_239351

/-- The total cost of the goods other than tuna and water -/
theorem total_cost_other_goods :
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  cost_other = 40 :=
by
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  show cost_other = 40
  sorry

end total_cost_other_goods_l239_239351


namespace probability_three_red_before_two_green_l239_239594

noncomputable def probability_red_chips_drawn_before_green (red_chips green_chips : ℕ) (total_chips : ℕ) : ℚ := sorry

theorem probability_three_red_before_two_green 
    (red_chips green_chips : ℕ) (total_chips : ℕ)
    (h_red : red_chips = 3) (h_green : green_chips = 2) 
    (h_total: total_chips = red_chips + green_chips) :
  probability_red_chips_drawn_before_green red_chips green_chips total_chips = 3 / 10 :=
  sorry

end probability_three_red_before_two_green_l239_239594


namespace circle_parabola_intersection_l239_239784

theorem circle_parabola_intersection (b : ℝ) :
  (∃ c r, ∀ x y : ℝ, y = (5 / 12) * x^2 → ((x - c)^2 + (y - b)^2 = r^2) ∧ 
   (y = (5 / 12) * x + b → ((x - c)^2 + (y - b)^2 = r^2))) → b = 169 / 60 :=
by
  sorry

end circle_parabola_intersection_l239_239784


namespace cost_of_fencing_rectangular_field_l239_239709

theorem cost_of_fencing_rectangular_field :
  (ratio : ℕ × ℕ) (area : ℕ) (cost_per_meter_paise : ℕ)
  (h_ratio : ratio = (3, 4))
  (h_area : area = 7500)
  (h_cost_paise : cost_per_meter_paise = 25) :
  (total_cost_rupees : ℚ) (h_total_cost : total_cost_rupees = 87.5) :=
sorry

end cost_of_fencing_rectangular_field_l239_239709


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239961

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239961


namespace expected_value_of_sum_of_marbles_l239_239119

theorem expected_value_of_sum_of_marbles :
  let marbles := [1, 2, 3, 4, 5, 6]
  let sets_of_three := Nat.choose 6 3  -- The number of ways to choose 3 marbles out of 6
  let all_sums := [
    1 + 2 + 3, 1 + 2 + 4, 1 + 2 + 5, 1 + 2 + 6, 1 + 3 + 4,
    1 + 3 + 5, 1 + 3 + 6, 1 + 4 + 5, 1 + 4 + 6, 1 + 5 + 6,
    2 + 3 + 4, 2 + 3 + 5, 2 + 3 + 6, 2 + 4 + 5, 2 + 4 + 6,
    2 + 5 + 6, 3 + 4 + 5, 3 + 4 + 6, 3 + 5 + 6, 4 + 5 + 6
  ]
  let total_sum := List.foldr (.+.) 0 all_sums
  let expected_value := total_sum / sets_of_three
  in expected_value = 10.5 := by
  sorry

end expected_value_of_sum_of_marbles_l239_239119


namespace bus_speed_in_kmph_l239_239316

-- Definition: Distance covered by the bus
def distance : ℝ := 600.048

-- Definition: Time taken by the bus
def time : ℝ := 30

-- Definition: Conversion factor from m/s to kmph
def conversion_factor : ℝ := 3.6

-- Definition: Speed in m/s
def speed_m_per_s : ℝ := distance / time

-- Theorem: The bus speed in kmph
theorem bus_speed_in_kmph : (speed_m_per_s * conversion_factor) ≈ 72.006 := by
  sorry

end bus_speed_in_kmph_l239_239316


namespace angle_double_l239_239156

variables {a : ℝ}
def point := (ℝ × ℝ)
def A : point := (0, 0)
def B : point := (a, 0)
def D : point := (0, a)
def C : point := (a, a)
def E : point := (a / 2, a)
def F : point := (3 * a / 4, a)

noncomputable def angle (p1 p2 p3 : point) : ℝ := sorry

theorem angle_double :
  angle B A F = 2 * angle D A E :=
sorry

end angle_double_l239_239156


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239960

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239960


namespace hyperbola_focal_length_l239_239464

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239464


namespace remainder_when_multiplied_and_divided_l239_239297

theorem remainder_when_multiplied_and_divided (n k : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := 
by
  sorry

end remainder_when_multiplied_and_divided_l239_239297


namespace part1_inequality_part2_range_of_a_l239_239527

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239527


namespace largest_beautiful_number_exists_l239_239339

def is_fair (m n : ℕ) (assignment : fin m -> fin n -> ℕ) : Prop :=
  (∑ i, ∑ j, assignment i j = (m * n) / 2) ∧ (∀ i j, assignment i j ∈ {0, 1})

def is_beautiful (a : ℝ) := 
  ∃ (m n : ℕ) (assignment : fin m -> fin n -> ℕ), 
  is_fair m n assignment ∧ 
  (∀ i : fin m, a <= 100 * (∑ j, assignment i j) / n ∧ 
              100 * (∑ j, assignment i j) / n <= 100 - a) ∧
  (∀ j : fin n, a <= 100 * (∑ i, assignment i j) / m ∧ 
              100 * (∑ i, assignment i j) / m <= 100 - a)

theorem largest_beautiful_number_exists : 
  ∃ (a : ℝ), 
  (is_beautiful 75) ∧ 
  (∀ b, is_beautiful b → b ≤ 75) :=
begin
  sorry
end

end largest_beautiful_number_exists_l239_239339


namespace minimum_distance_midpoint_to_origin_l239_239129

def line (A B C : ℝ) (P : ℝ × ℝ) : Prop :=
  A * P.1 + B * P.2 + C = 0

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem minimum_distance_midpoint_to_origin :
  ∃ (A B : ℝ × ℝ), 
    line 1 1 (-7) A ∧
    line 1 1 (-5) B ∧
    distance (midpoint A B) (0, 0) = 3 * Real.sqrt 2 :=
by
  sorry

end minimum_distance_midpoint_to_origin_l239_239129


namespace smallest_solution_l239_239040

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239040


namespace geometric_series_sum_l239_239365

theorem geometric_series_sum :
  let a := -1
  let r := -2
  let n := 10
  let S_n := a * (r ^ n - 1) / (r - 1)
  S_n = 341 := by
  let a := -1
  let r := -2
  let n := 10
  let S_n := a * (r ^ n - 1) / (r - 1)
  exact Eq.refl S_n
  sorry

end geometric_series_sum_l239_239365


namespace sum_of_roots_l239_239279

theorem sum_of_roots : 
  let f := λ x : ℝ, x^3 - 3 * x^2 - 12 * x - 7 in 
  let roots_sum := 3 in 
  (∀ x, f x = 0 → ∃ y, (y = 3)) :=
sorry

end sum_of_roots_l239_239279


namespace hyperbola_focal_length_proof_l239_239455

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239455


namespace quadratic_equation_root_l239_239384

def has_rational_coefficients (a b c : ℚ) := 
  ∃ p q r : ℤ, a = p ∧ b = q ∧ c = r

theorem quadratic_equation_root (a b c : ℚ) (h_rational : has_rational_coefficients a b c)
  (h_quad : a = 1) (h_root : Polynomial.eval (Real.sqrt 5 - 3) (Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c) = 0) :
  a = 1 ∧ b = 6 ∧ c = -4 :=
by
  sorry

end quadratic_equation_root_l239_239384


namespace problem1_problem2_l239_239521

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239521


namespace license_plate_possibilities_l239_239347

-- Definitions of conditions
def license_plate_combinations (letters : List Char) (digits : List Nat) : Nat :=
  if (letters.length = 4 ∧ digits.length = 3 ∧
      (letters.count_occurrences 'A' = 2 ∨ letters.count_occurrences 'B' = 2 ∨ 
       letters.count_occurrences 'C' = 2 ∨ -- additional checks for all 26 letters
       letters.count_occurrences 'Z' = 2) ∧
      (list.pairwise (≤) digits)) then
    let choose_2_out_of_26 := 26 * (26 - 1) / 2
    let arrange_letters := 4 * (4 - 1) / 2
    let choose_digits := 8
    choose_2_out_of_26 * arrange_letters * choose_digits
  else 0

#eval license_plate_combinations ['A', 'A', 'B', 'B'] [2, 3, 4]  -- should return 15600

theorem license_plate_possibilities : license_plate_combinations ['A', 'A', 'B', 'B'] [0, 1, 2] = 15600 :=
  sorry

end license_plate_possibilities_l239_239347


namespace probability_one_absent_l239_239995

theorem probability_one_absent (p_absent : ℚ)
    (p_present : ℚ) : 
    let prob := 3 * (p_absent * (p_present)^2) in
    p_absent = 1/20 →
    p_present = 19/20 →
    prob * 100 = 13.5 :=
by
  intro h1 h2
  sorry

end probability_one_absent_l239_239995


namespace find_m_range_l239_239081

noncomputable def has_distinct_negative_real_roots (m : ℝ) : Prop :=
  let Δ := m^2 - 4 in
  Δ > 0 ∧ -m < 0 ∧ 1 > 0

noncomputable def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem find_m_range : {m : ℝ | (has_distinct_negative_real_roots m ∨ inequality_holds m) ∧ ¬ (has_distinct_negative_real_roots m ∧ inequality_holds m)} = {m : ℝ | (1 < m ∧ m ≤ 2) ∨ (3 ≤ m)} :=
by 
  sorry

end find_m_range_l239_239081


namespace find_f_of_5_l239_239512

def f (x : ℕ) : ℕ :=
  if x ≥ 6 then x - 3 else f (f (x + 5))

theorem find_f_of_5 : f 5 = 4 :=
  sorry

end find_f_of_5_l239_239512


namespace find_number_l239_239394

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l239_239394


namespace candle_height_relation_l239_239723

variables (t : ℝ)

def height_candle_A (t : ℝ) := 12 - 2 * t
def height_candle_B (t : ℝ) := 9 - 2 * t

theorem candle_height_relation : 
  12 - 2 * (15 / 4) = 3 * (9 - 2 * (15 / 4)) :=
by
  sorry

end candle_height_relation_l239_239723


namespace all_a_n_are_perfect_squares_l239_239898

noncomputable def c : ℕ → ℤ 
| 0 => 1
| 1 => 0
| 2 => 2005
| n+2 => -3 * c n - 4 * c (n-1) + 2008

noncomputable def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4 ^ n * 2004 * 501

theorem all_a_n_are_perfect_squares (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 :=
by
  sorry

end all_a_n_are_perfect_squares_l239_239898


namespace focal_length_of_hyperbola_l239_239485

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239485


namespace solve_eq1_solve_eq2_l239_239663

theorem solve_eq1 (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
by
  sorry

end solve_eq1_solve_eq2_l239_239663


namespace number_of_three_digit_integers_with_strictly_increasing_odd_digits_l239_239117

/-
  Prove that there are exactly 10 three-digit integers such that:
  - Their digits, read left to right, are in strictly increasing order.
  - All digits are odd.
-/
theorem number_of_three_digit_integers_with_strictly_increasing_odd_digits :
  ∃! num : ℕ, num = 10 ∧ (∀ (a b c : ℕ), 
    (a ∈ {1, 3, 5, 7, 9} ∧ b ∈ {1, 3, 5, 7, 9} ∧ c ∈ {1, 3, 5, 7, 9} ∧ a < b ∧ b < c) →
    ∃ n : ℕ, n = 100 * a + 10 * b + c) :=
begin
  sorry,
end

end number_of_three_digit_integers_with_strictly_increasing_odd_digits_l239_239117


namespace point_in_first_quadrant_l239_239868

noncomputable def quadrant_of_complex (z : ℂ) (hz : z * (4 + complex.I) = 3 + complex.I) : Prop :=
  (z.re > 0) ∧ (z.im > 0)

theorem point_in_first_quadrant (z : ℂ) (hz : z * (4 + complex.I) = 3 + complex.I) : quadrant_of_complex z hz :=
sorry

end point_in_first_quadrant_l239_239868


namespace trigonometric_expression_proof_l239_239374

theorem trigonometric_expression_proof :
  (Real.cos (76 * Real.pi / 180) * Real.cos (16 * Real.pi / 180) +
   Real.cos (14 * Real.pi / 180) * Real.cos (74 * Real.pi / 180) -
   2 * Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)) = 0 :=
by
  sorry

end trigonometric_expression_proof_l239_239374


namespace Tyler_age_l239_239176

variable (T B S : ℕ)

theorem Tyler_age :
  (T = B - 3) ∧
  (S = B + 2) ∧
  (S = 2 * T) ∧
  (T + B + S = 30) →
  T = 5 := by
  sorry

end Tyler_age_l239_239176


namespace minimize_ab_value_l239_239915

noncomputable def minimize (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 9 * a + b = 36) : ℝ :=
  a * b

theorem minimize_ab_value : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 9 * a + b = 36 ∧ minimize a b sorry sorry sorry = 36 :=
sorry

end minimize_ab_value_l239_239915


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239957

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239957


namespace range_of_m_l239_239368

-- Definitions of the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x + 1| > m
def q (m : ℝ) : Prop := ∀ x > 2, 2 * x - 2 * m > 0

-- The main theorem statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l239_239368


namespace isosceles_right_triangle_ratio_l239_239340

theorem isosceles_right_triangle_ratio {a : ℝ} (h_pos : 0 < a) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 :=
sorry

end isosceles_right_triangle_ratio_l239_239340


namespace part1_part2_part3_l239_239501

-- Defining the conditions given in the problem
def is_equilateral_triangle (A B O : Point) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O A ∧ dist O A = 2 * sqrt 3

def parabola_equation (C1 : Point → Prop) : Prop :=
  ∃ c, C1 = { P : Point | P.2^2 = 4 * c * P.1 }

-- Given conditions in Lean 4
variables {a b : ℝ} (h1 : 0 < b) (h2 : b < a)
def ellipse (P : Point → Prop) : Prop :=
  P = { P : Point | P.x^2 / a^2 + P.y^2 / b^2 = 1 }

def focus (F : Point) : Prop :=
  F = (sqrt (a^2 - b^2), 0)

noncomputable def C1 (P : Point) : Prop :=
  P = (x, y) => y^2 = x

def vertex (O : Point) : Prop :=
  O = (0, 0)

def intersection (A B : Point) (C1 C2 : Point → Prop) : Prop :=
  C1 A ∧ C2 A ∧ C1 B ∧ C2 B ∧ A.y > 0 ∧ B.y < 0

-- Main proof statements in Lean 4
theorem part1 (A B : Point) (AOB_eq_tri : is_equilateral_triangle A B O) (intersect : intersection A B C1 C2) : parabola_equation C1 :=
  sorry

theorem part2 (c : ℝ) (eccentricity : ℝ) (AF_perp_OF : AF ⊥ OF) : eccentricity = sqrt 2 - 1 :=
  sorry

theorem part3 (P : Point) (intersect_MN : ∃ M N : Point, (M.x = m ∧ M.y = 0) ∧ (N.x = n ∧ N.y = 0) ∧ lines_intersect AP BP M N) : m * n = a^2 :=
  sorry

end part1_part2_part3_l239_239501


namespace school_committee_count_l239_239405

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def valid_committees_count (total_students total_teachers committee_size : ℕ) : ℕ :=
  let total_people := total_students + total_teachers
  let total_combinations := binom total_people committee_size
  let student_only_combinations := binom total_students committee_size
  total_combinations - student_only_combinations

theorem school_committee_count :
  valid_committees_count 12 3 9 = 4785 :=
by {
  -- Translate the calculation described in the problem to a Lean statement.
  let total_combinations := binom 15 9,
  let student_only_combinations := binom 12 9,
  let valid_com := total_combinations - student_only_combinations,

  -- General binomial coefficient computation simplification is omitted.
  -- Simplify the exact computation here using known binomial identities as required.
  have h1 : binom 15 9 = 5005 := sorry,
  have h2 : binom 12 9 = 220 := sorry,
  
  -- Valid committee count check
  have h3: valid_com = 5005 - 220 := sorry,
  have h4: valid_com = 4785 := by norm_num,
  exact h4,
}

end school_committee_count_l239_239405


namespace profit_percentage_l239_239296

-- Definitions and conditions
variable (SP : ℝ) (CP : ℝ)
variable (h : CP = 0.98 * SP)

-- Lean statement to prove the profit percentage is 2.04%
theorem profit_percentage (h : CP = 0.98 * SP) : (SP - CP) / CP * 100 = 2.04 := 
sorry

end profit_percentage_l239_239296


namespace smallest_n_l239_239171

def sixtyoned (n : ℕ) : ℕ :=
  (finset.range (n+1)).sum (λ x, ((finset.range (n+1)).filter (λ y, ((x + 1)^2 - x * y * (2 * x - x * y + 2 * y) + (y + 1)^2 = n))).card)

theorem smallest_n (n : ℕ) (h : ∀ m < n, sixtyoned m < 61) : ∃ n, sixtyoned n = 61 → n = 2^120 - 2^61 + 2 :=
by sorry

end smallest_n_l239_239171


namespace problem1_problem2_l239_239525

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239525


namespace sin_double_angle_fourth_quadrant_l239_239972

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239972


namespace polynomial_g_is_given_value_l239_239582

open Polynomial

theorem polynomial_g_is_given_value :
  ∀ (f g : Polynomial ℝ),
    (f + g = C 2 * X^2 - C 3) ∧ (f = X^4 - C 3 * X^2 + C 1) →
    g = -X^4 + C 5 * X^2 - C 4 :=
by
  intros f g h
  cases h with h1 h2
  rw [h2, add_comm g f] at h1
  rw [add_sub_cancel'_right] at h1
  exact h1

end polynomial_g_is_given_value_l239_239582


namespace number_of_valid_subsets_l239_239190

def S : Set ℕ := {0, 1, 2, 3, 4, 5}

def is_isolated_element (A : Set ℕ) (x : ℕ) : Prop := x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

def no_isolated_elements (A : Set ℕ) : Prop := ∀ x, x ∈ A → ¬ is_isolated_element A x

def four_element_subsets (S : Set ℕ) : Set (Set ℕ) := { A | A ⊆ S ∧ A.card = 4 }

def valid_subsets (S : Set ℕ) : Set (Set ℕ) :=
  { A | A ∈ four_element_subsets S ∧ no_isolated_elements A }

theorem number_of_valid_subsets : ∃ (n : ℕ), n = 6 ∧ n = Set.card (valid_subsets S) :=
by {
  sorry
}

end number_of_valid_subsets_l239_239190


namespace max_vertices_no_rectangle_l239_239741

/-- In a regular 2016-sided polygon (2016-gon), prove that the maximum number of vertices 
    that can be marked such that no four marked vertices form the vertices of a rectangle is 1009. -/
theorem max_vertices_no_rectangle (n : ℕ) (h : n = 2016) : 
  ∃ (m : ℕ), m = 1009 ∧ 
    ∀ (marked : finset (fin n)), 
      marked.card ≤ m → 
      (¬ ∃ (a b c d : fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
        (is_rectangle a b c d)) := 
begin 
  sorry 
end

/-- Predicate to determine if four vertices form a rectangle in a regular n-gon. -/
def is_rectangle (a b c d : fin 2016) : Prop := 
  ∃ (k : ℕ), k ∈ finset.range 1008 ∧ 
    ((a = fin.of_nat k) ∧ (b = fin.of_nat (k + 1008)) ∧ 
     (c = fin.of_nat (k + 1008 + 1)) ∧ (d = fin.of_nat (k + 1)) ∨ 
     (a = fin.of_nat (k + 1008)) ∧ (b = fin.of_nat k) ∧ 
     (c = fin.of_nat (k + 1)) ∧ (d = fin.of_nat (k + 1008 + 1)))

end max_vertices_no_rectangle_l239_239741


namespace perpendicular_lines_l239_239697

theorem perpendicular_lines (a : ℝ) :
  let line1_slope := - (1 / a),
      line2_slope := (a + 1) / -2 in
  (line1_slope * line2_slope = -1) ↔ a = -1 :=
sorry

end perpendicular_lines_l239_239697


namespace coeff_x2_30_l239_239172

noncomputable def coeff (p : Polynomial ℚ) (n : ℕ) : ℚ := p.coeff n

theorem coeff_x2_30 : 
  let p := (1 - Polynomial.X) * (1 + 2 * Polynomial.X)^5 
  in coeff p 2 = 30 :=
by
  sorry

end coeff_x2_30_l239_239172


namespace hyperbola_focal_length_l239_239475

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239475


namespace min_area_triangle_l239_239642

-- Let point \(C\) be a moving point on the parabola \(y^2 = 2x\)
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Circle equation \((x - 1)^2 + y^2 = 1\)
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- The intersection points \(A\) and \(B\) are given by the tangent lines from \(C\) to circle
def tangent_line (C : ℝ × ℝ) (x y : ℝ) : Prop := 
    ∃ k : ℝ, y = k * x + 2 * C.snd * (1 - k * C.snd)

-- Condition that point \(C\) intersects parabola
variable (C : ℝ × ℝ)
variable (intersects_parabola : parabola C.fst C.snd)

-- Condition that lines from \(C\) are tangents to the circle
variable (A B : ℝ × ℝ)
variable (tangentA : tangent_line C A.fst A.snd)
variable (tangentB : tangent_line C B.fst B.snd)
variable (on_circleA : circle A.fst A.snd)
variable (on_circleB : circle B.fst B.snd)

-- Proof problem: minimum area of triangle ABC
theorem min_area_triangle (h : ∃ A B : ℝ × ℝ, tangent_line C A.fst A.snd ∧ tangent_line C B.fst B.snd ∧ circle A.fst A.snd ∧ circle B.fst B.snd) :
  ∃ A B C : ℝ × ℝ, parabola C.fst C.snd ∧ tangent_line C A.fst A.snd ∧ tangent_line C B.fst B.snd ∧ circle A.fst A.snd ∧ circle B.fst B.snd ∧ (AreaABC A B C = 8) :=
sorry

end min_area_triangle_l239_239642


namespace relationship_B_not_correlation_l239_239287

def is_correlation (relationship : String) : Prop := sorry

def relationship_A := 'The relationship between rice yield in a field and fertilization'
def relationship_B := 'The relationship between the area of a square and its side length'
def relationship_C := 'The relationship between sales revenue of a product and its advertising expenses'
def relationship_D := 'The relationship between body fat content and age'

theorem relationship_B_not_correlation
  (h_A : is_correlation relationship_A)
  (h_C : is_correlation relationship_C)
  (h_D : is_correlation relationship_D) :
  ¬ is_correlation relationship_B := sorry

end relationship_B_not_correlation_l239_239287


namespace find_initial_population_l239_239704

-- Define the conditions that the population increases annually by 20%
-- and that the population after 2 years is 14400.
def initial_population (P : ℝ) : Prop :=
  1.44 * P = 14400

-- The theorem states that given the conditions, the initial population is 10000.
theorem find_initial_population (P : ℝ) (h : initial_population P) : P = 10000 :=
  sorry

end find_initial_population_l239_239704


namespace inverse_function_condition_l239_239622

noncomputable def f (m x : ℝ) := (3 * x + 4) / (m * x - 5)

theorem inverse_function_condition (m : ℝ) :
  (∀ x : ℝ, f m (f m x) = x) ↔ m = -4 / 5 :=
by
  sorry

end inverse_function_condition_l239_239622


namespace sum_of_b_values_for_one_solution_l239_239845

theorem sum_of_b_values_for_one_solution :
  (∀ b : ℝ, ∃ x : ℝ, 3 * x^2 + b * x + 6 * x + 4 = 0 ∧
  (∀ y : ℝ, (3 * y^2 + b * y + 6 * y + 4 = 0 → y = x)) →
  ((b = -6 + 4 * sqrt 3 ∨ b = -6 - 4 * sqrt 3) → (b + b) = -12)) :=
sorry

end sum_of_b_values_for_one_solution_l239_239845


namespace range_of_m_l239_239991

theorem range_of_m (m x : ℝ) (h : x ∈ Icc 0 2) (root : x^3 - 3 * x - m = 0) : 
  m ∈ Icc (-2 : ℝ) 2 :=
sorry

end range_of_m_l239_239991


namespace smallest_solution_to_equation_l239_239035

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239035


namespace sequence_general_term_l239_239109

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term_l239_239109


namespace find_xyz_l239_239904

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := 
sorry

end find_xyz_l239_239904


namespace number_of_planes_parallel_to_l_l239_239930

axiom line : Type
axiom point : Type
axiom lies_on (p : point) (l : line) : Prop
axiom plane (l : line) : Type
axiom parallel_to (p : plane l) (l : line) : Prop

variables (A B : point) (l : line)
axiom not_lies_on_A : ¬ lies_on A l
axiom not_lies_on_B : ¬ lies_on B l

theorem number_of_planes_parallel_to_l (h : A ≠ B) : 
  ∃ n, (n = 0 ∨ n = 1 ∨ n = ℵ₀) ∧
      n = ∃ p : plane l, parallel_to p l := 
sorry

end number_of_planes_parallel_to_l_l239_239930


namespace Mary_forgot_pigs_l239_239647

theorem Mary_forgot_pigs (Mary_thinks : ℕ) (actual_animals : ℕ) (double_counted_sheep : ℕ)
  (H_thinks : Mary_thinks = 60) (H_actual : actual_animals = 56)
  (H_double_counted : double_counted_sheep = 7) :
  ∃ pigs_forgot : ℕ, pigs_forgot = 3 :=
by
  let counted_animals := Mary_thinks - double_counted_sheep
  have H_counted_correct : counted_animals = 53 := by sorry -- 60 - 7 = 53
  have pigs_forgot := actual_animals - counted_animals
  have H_pigs_forgot : pigs_forgot = 3 := by sorry -- 56 - 53 = 3
  exact ⟨pigs_forgot, H_pigs_forgot⟩

end Mary_forgot_pigs_l239_239647


namespace science_books_initially_l239_239244

def initial_number_of_books (borrowed left : ℕ) : ℕ := 
borrowed + left

theorem science_books_initially (borrowed left : ℕ) (h1 : borrowed = 18) (h2 : left = 57) :
initial_number_of_books borrowed left = 75 := by
sorry

end science_books_initially_l239_239244


namespace money_left_is_40_l239_239201

-- Define the quantities spent by Mildred and Candice, and the total money provided by their mom.
def MildredSpent : ℕ := 25
def CandiceSpent : ℕ := 35
def TotalGiven : ℕ := 100

-- Calculate the total spent and the remaining money.
def TotalSpent := MildredSpent + CandiceSpent
def MoneyLeft := TotalGiven - TotalSpent

-- Prove that the money left is 40.
theorem money_left_is_40 : MoneyLeft = 40 := by
  sorry

end money_left_is_40_l239_239201


namespace arcsin_arccos_unit_circle_l239_239238

/-- The equation arcsin(x) + arccos(y) = n * π (with n ∈ ℤ) represents parts of the unit circle.
Specifically:
1. For n = 0, it represents the part of the unit circle where x ≤ 0 and y ≥ 0 (second quadrant).
2. For n = 1, it represents the part of the unit circle where x ≥ 0 and y ≤ 0 (fourth quadrant).
-/
theorem arcsin_arccos_unit_circle (x y : ℝ) (n : ℤ) :
  (arcsin x + arccos y = n * ℝ.pi) →
  (x^2 + y^2 = 1) ∧
  ((n = 0 → x ≤ 0 ∧ y ≥ 0) ∧ (n = 1 → x ≥ 0 ∧ y ≤ 0)) :=
by
  sorry

end arcsin_arccos_unit_circle_l239_239238


namespace projectiles_meet_in_84_minutes_l239_239299

-- Define the variables and constants
def distance : ℝ := 1386
def speed1 : ℝ := 445
def speed2 : ℝ := 545

-- Calculate the combined speed
def combined_speed : ℝ := speed1 + speed2

-- Calculate the time in hours to meet
def time_to_meet_in_hours : ℝ := distance / combined_speed

-- Convert time to minutes
def time_to_meet_in_minutes : ℝ := time_to_meet_in_hours * 60

-- Statement of the problem
theorem projectiles_meet_in_84_minutes :
  time_to_meet_in_minutes = 84 :=
by sorry

end projectiles_meet_in_84_minutes_l239_239299


namespace part1_part2_l239_239541

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239541


namespace cut_and_reassemble_squares_l239_239837

theorem cut_and_reassemble_squares (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let s1 := a^2
      s2 := b^2
      s3 := (real.sqrt (a^2 + b^2))^2 in
  s1 + s2 = s3 := 
begin
  sorry
end

end cut_and_reassemble_squares_l239_239837


namespace neznaika_discrepancy_l239_239289

-- Definitions of the given conditions
def correct_kilograms_to_kilolunas (kg : ℝ) : ℝ := kg / 0.24
def neznaika_kilograms_to_kilolunas (kg : ℝ) : ℝ := (kg * 4) * 1.04

-- Define the precise value of 1 kiloluna in kilograms for clarity
def one_kiloluna_in_kg : ℝ := (1 / 4) * 0.96

-- Function calculating the discrepancy percentage
def discrepancy_percentage (kg : ℝ) : ℝ :=
  let correct_value := correct_kilograms_to_kilolunas kg
  let neznaika_value := neznaika_kilograms_to_kilolunas kg
  ((correct_value - neznaika_value).abs / correct_value) * 100

-- Statement of the theorem to be proven
theorem neznaika_discrepancy :
  discrepancy_percentage 1 = 0.16 := sorry

end neznaika_discrepancy_l239_239289


namespace dogs_distribution_impossible_l239_239208

theorem dogs_distribution_impossible :
  ∀ (labs goldens shepherds bulldogs beagles poodles rottweilers : ℕ)
    (doghouses total_dogs : ℕ)
    (dogs_per_house : ℕ → Prop),
  labs = 8 ∧ goldens = 8 ∧ shepherds = 10 ∧ bulldogs = 6 ∧ beagles = 6 ∧ poodles = 6 ∧ rottweilers = 6 ∧
  doghouses = 10 ∧ total_dogs = 50 ∧
  (∀ n, dogs_per_house n → n = 4 ∨ n = 6) →
  (shepherds = 2 * 4 ∨ shepherds = 4 + 6) → -- German Shepherds need 10 (4+6 or 2*4)
  (bulldogs = 6) ∧ (beagles = 6) ∧ (poodles = 6) ∧ (rottweilers = 6) ∧ -- 1 house each
  (labs = 2 * 4) ∧ (goldens = 2 * 4) ∧ -- Labs and Golden Retrievers 4+4 in two separate houses each
  ¬ ∃ assignment : list (list ℕ),
    (∀ house, house ∈ assignment → list.sum house = 4 ∨ list.sum house = 6) ∧
    list.length assignment = doghouses :=
sorry

end dogs_distribution_impossible_l239_239208


namespace min_dot_product_PF1_PF2_parallelogram_condition_l239_239889

-- Define the ellipse equation
def on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 2) = 1

-- Define the dot product function between PF1 and PF2
def dot_product_PF1_PF2 (x₀ y₀ : ℝ) : ℝ :=
  let PF1 := (-√3 - x₀, -y₀)
  let PF2 := (√3 - x₀, -y₀)
  (PF1.1 * PF2.1) + (PF1.2 * PF2.2)

-- Prove that the min value of the dot product is -8/3
theorem min_dot_product_PF1_PF2 :
  ∀ x₀ y₀, on_ellipse x₀ y₀ → 
  ∃ x_min : ℝ, x_min = -√3 ∨ x_min = √3 ∧
  min (dot_product_PF1_PF2 x₀ y₀) = -8 / 3 :=
by
  intro x₀ y₀ h_on_ellipse
  sorry

-- Define the necessary conditions and proofs for part II
theorem parallelogram_condition (x₀ y₀ : ℝ) (k : ℝ) 
   (P Q A B : ℝ × ℝ) (h_on_ellipse : on_ellipse x₀ y₀) 
   (h_y₀_pos : y₀ > 0) (h_dot_product_eq_zero : dot_product_PF1_PF2 x₀ y₀ = 0) : 
  ∃ l_eq : ℝ → ℝ,
  l_eq = λ x, - (sqrt 3 / 3) * (x + 1) ∧
  (PABQ_parallelogram : quadrilateral_parallelogram P A B Q) :=
by
  intro x₀ y₀ k P Q A B h_on_ellipse h_y₀_pos h_dot_product_eq_zero
  sorry

end min_dot_product_PF1_PF2_parallelogram_condition_l239_239889


namespace cube_plane_intersection_diff_l239_239847

theorem cube_plane_intersection_diff (Q : cube) (p : ℕ → plane) (S : set face) (hS : S = ⋃ (f : face) (H : f ∈ faces Q), {f})
  (P : set point) (hP : P = ⋃ (j : ℕ) (H : j ≤ k), points_of_plane (p j))
  (h_intersect : ∀ (f : face), intersect_plane_face (P, f) = segments face_center_to_vertices f):
  max_k = 6 ∧ min_k = 6 → max_k - min_k = 0 :=
begin
  sorry
end

end cube_plane_intersection_diff_l239_239847


namespace carrot_cakes_in_february_l239_239676

theorem carrot_cakes_in_february :
  ∀ (oct nov dec jan : ℕ), oct = 19 → nov = 21 → dec = 23 → jan = 25 →
  (∀ m n : ℕ, n = m + 2) → jan + 2 = 27 :=
by
  -- Conditions derived from the problem
  intros oct nov dec jan H_oct H_nov H_dec H_jan pattern
  -- Using the conditions to prove the statement
  rw H_jan
  -- Applying the pattern
  exact pattern jan 27 sorry -- Sorry is used to skip the proof.

end carrot_cakes_in_february_l239_239676


namespace part1_inequality_part2_range_of_a_l239_239533

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l239_239533


namespace part1_solution_set_part2_range_of_a_l239_239537

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239537


namespace smallest_solution_l239_239060

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239060


namespace number_of_triangles_lower_bound_l239_239413

open set finset nat int real

theorem number_of_triangles_lower_bound
  (n : ℕ) (m : ℕ) (h1 : n > 3) 
  (h2 : ∀ p1 p2 p3 : fin n, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear ({p1, p2, p3} : finset (fin n)))
  (segments : finset (fin n × fin n)) 
  (h3 : segments.card = m) :
  ∃ t : ℕ, t ≥ (m * (4 * m - n * n)) / (3 * n) :=
begin
  sorry
end

end number_of_triangles_lower_bound_l239_239413


namespace smallest_solution_l239_239045

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239045


namespace sin_double_angle_neg_l239_239947

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239947


namespace hyperbola_focal_length_l239_239477

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239477


namespace regular_octagon_opposite_sides_eq_l239_239820

theorem regular_octagon_opposite_sides_eq (a b c d e f g h : ℤ) 
  (h_equal_angles : true) 
  (h_int_sides : true) 
  (h_sides : List.nth [a, b, c, d, e, f, g, h] 0 = Option.some a ∧
             List.nth [a, b, c, d, e, f, g, h] 1 = Option.some b ∧
             List.nth [a, b, c, d, e, f, g, h] 2 = Option.some c ∧
             List.nth [a, b, c, d, e, f, g, h] 3 = Option.some d ∧
             List.nth [a, b, c, d, e, f, g, h] 4 = Option.some e ∧
             List.nth [a, b, c, d, e, f, g, h] 5 = Option.some f ∧
             List.nth [a, b, c, d, e, f, g, h] 6 = Option.some g ∧
             List.nth [a, b, c, d, e, f, g, h] 7 = Option.some h) :
  a = e ∧ b = f ∧ c = g ∧ d = h :=
sorry

end regular_octagon_opposite_sides_eq_l239_239820


namespace sequence_conjecture_l239_239897

theorem sequence_conjecture (a : ℕ+ → ℕ) (S : ℕ+ → ℕ) (h : ∀ n : ℕ+, 2 * S n = 4 * a n + (n - 4) * (n + 1)) :
  ∀ n : ℕ+, a n = 2 ^ n + n := 
by
  sorry

end sequence_conjecture_l239_239897


namespace num_ways_seating_l239_239150

theorem num_ways_seating (n : ℕ) (h : n = 6) : (nat.factorial n) / n = nat.factorial (n - 1) :=
by 
  rw h
  calc
    (nat.factorial 6) / 6 = 720 / 6    : by norm_num
                      ... = 120        : by norm_num
                      ... = nat.factorial 5 : by norm_num

end num_ways_seating_l239_239150


namespace Liz_latest_start_time_l239_239197

noncomputable def latest_start_time (turkey_weight : ℕ) (roast_time_per_pound : ℕ) (number_of_turkeys : ℕ) (dinner_time : Time) : Time :=
  Time.sub dinner_time (
    ((turkey_weight * roast_time_per_pound) * number_of_turkeys) / 60
  )

theorem Liz_latest_start_time : 
  latest_start_time 16 15 2 (Time.mk 18 0) = Time.mk 10 0 := 
by
  sorry

end Liz_latest_start_time_l239_239197


namespace alcohol_solution_mixing_l239_239311

theorem alcohol_solution_mixing :
  ∀ (V_i C_i C_f C_a x : ℝ),
    V_i = 6 →
    C_i = 0.40 →
    C_f = 0.50 →
    C_a = 0.90 →
    x = 1.5 →
  0.50 * (V_i + x) = (C_i * V_i) + C_a * x →
  C_f * (V_i + x) = (C_i * V_i) + (C_a * x) := 
by
  intros V_i C_i C_f C_a x Vi_eq Ci_eq Cf_eq Ca_eq x_eq h
  sorry

end alcohol_solution_mixing_l239_239311


namespace set_B_correct_l239_239431

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l239_239431


namespace vector_dot_product_range_l239_239905

variables {V : Type} [inner_product_space ℝ V]
variables (A B C P O : V) (l : ℝ)
variables (hA : (A - O).norm = l) (hB : (B - O).norm = l) (hC : (C - O).norm = l)
          (hAB : A - O = - (B - O))
          (hP : (P - O).norm ≤ l)

theorem vector_dot_product_range :
  let PA := P - A in
  let PB := P - B in
  let PC := P - C in
  (- (4 / 3) * l^2 ≤ (PA ⬝ PB + PB ⬝ PC + PC ⬝ PA) ∧ (PA ⬝ PB + PB ⬝ PC + PC ⬝ PA) ≤ 4 * l^2) :=
sorry

end vector_dot_product_range_l239_239905


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239967

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239967


namespace balloon_altitude_l239_239874

theorem balloon_altitude 
  (temp_diff_per_1000m : ℝ)
  (altitude_temp : ℝ) 
  (ground_temp : ℝ)
  (altitude : ℝ) 
  (h1 : temp_diff_per_1000m = 6) 
  (h2 : altitude_temp = -2)
  (h3 : ground_temp = 5) :
  altitude = 7/6 :=
by sorry

end balloon_altitude_l239_239874


namespace max_x_value_l239_239674

noncomputable def max_x (y : ℕ) (prime_y : Nat.Prime y) : ℕ :=
  if h : y > 1 then
    let upper_bound := Nat.floor (Real.sqrt ((800000 - 4.26) * y^3 / 2.75))
    (List.filter (fun x => Nat.gcd x y = 1) (List.range (upper_bound + 1))).reverse.head
  else 0

theorem max_x_value : 
  (∀ x y : ℕ, y > 1 → Nat.Prime y → (2.75 * x^2) / y^3 + 4.26 < 800000 → Nat.gcd x y = 1 → x + y is minimized)
  → max_x 3 (by norm_num) = 2801 :=
by sorry

end max_x_value_l239_239674


namespace compound_interest_l239_239298

theorem compound_interest (SI : ℝ) (P : ℝ) (R : ℝ) (T : ℝ) (CI : ℝ) :
  SI = 50 →
  R = 5 →
  T = 2 →
  P = (SI * 100) / (R * T) →
  CI = P * (1 + R / 100)^T - P →
  CI = 51.25 :=
by
  intros
  exact sorry -- This placeholder represents the proof that would need to be filled in 

end compound_interest_l239_239298


namespace product_b6_b8_is_16_l239_239892

-- Given conditions
variable (a : ℕ → ℝ) -- Sequence a_n
variable (b : ℕ → ℝ) -- Sequence b_n

-- Condition 1: Arithmetic sequence a_n and non-zero
axiom a_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_non_zero : ∃ n, a n ≠ 0

-- Condition 2: Equation 2a_3 - a_7^2 + 2a_n = 0
axiom a_satisfies_eq : ∀ n : ℕ, 2 * a 3 - (a 7) ^ 2 + 2 * a n = 0

-- Condition 3: Geometric sequence b_n with b_7 = a_7
axiom b_is_geometric : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n
axiom b7_equals_a7 : b 7 = a 7

-- Prove statement
theorem product_b6_b8_is_16 : b 6 * b 8 = 16 := sorry

end product_b6_b8_is_16_l239_239892


namespace largest_increase_year_l239_239366

-- The yearly profits from 2000 to 2010
def profits : List (ℕ × ℝ) :=
  [(2000, 2.0), (2001, 2.4), (2002, 3.0), (2003, 3.5), (2004, 4.5),
   (2005, 4.8), (2006, 5.3), (2007, 5.0), (2008, 3.5), (2009, 4.0), (2010, 3.0)]

-- The statement that needs to be proven
theorem largest_increase_year : 
  (∀ (profits : List (ℕ × ℝ)),  
    let increases := List.map (λ p : (ℕ × ℝ) × (ℕ × ℝ), (p.2.1, p.2.2 - p.1.2)) 
                     (List.zip profits (List.drop 1 profits)) in
    let max_increase := List.maximumBy (λ p1 p2, Real.lt (p1.2) (p2.2)) increases in
    max_increase.1 = 2004) :=
sorry

end largest_increase_year_l239_239366


namespace math_problem_l239_239719

theorem math_problem
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2006)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2007)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2006)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2007)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2006)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2007)
  : (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = -1 / 2006 := by
  sorry

end math_problem_l239_239719


namespace max_marked_vertices_no_rectangle_l239_239742

theorem max_marked_vertices_no_rectangle (n : ℕ) (hn : n = 2016) : 
  ∃ m ≤ n, m = 1009 ∧ 
  ∀ A B C D : Fin n, 
    (A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) ∧ 
    (marked A → marked B → marked C → marked D → 
     ¬is_rectangle A B C D) → 
      (∃ f : Fin n → Bool, marked f ∧ 
      (count_marked f ≤ 1009)) := sorry

end max_marked_vertices_no_rectangle_l239_239742


namespace neither_cakes_nor_cookies_l239_239795

-- Define the share of sales for cakes and cookies
def cake_sales : ℕ := 42
def cookie_sales : ℕ := 25

-- Define the condition that the total sales is 100%
def total_sales : ℕ := 100

-- Prove that the percentage of sales that were neither cakes nor cookies is 33%
theorem neither_cakes_nor_cookies {cake_sales cookie_sales total_sales: ℕ} :
  cake_sales = 42 →
  cookie_sales = 25 →
  total_sales = 100 →
  (total_sales - (cake_sales + cookie_sales)) = 33 :=
by
  intros h_cake h_cookie h_total
  rw [h_cake, h_cookie, h_total]
  norm_num
  sorry

end neither_cakes_nor_cookies_l239_239795


namespace Olly_needs_24_shoes_l239_239205

def dogs := 3
def cats := 2
def ferrets := 1
def paws_per_dog := 4
def paws_per_cat := 4
def paws_per_ferret := 4

theorem Olly_needs_24_shoes : (dogs * paws_per_dog) + (cats * paws_per_cat) + (ferrets * paws_per_ferret) = 24 :=
by
  sorry

end Olly_needs_24_shoes_l239_239205


namespace sin_double_angle_fourth_quadrant_l239_239974

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239974


namespace lean_proof_l239_239912

variables (A B C D E F : Type)
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables [inner_product_space ℝ D] [inner_product_space ℝ E] [inner_product_space ℝ F]
variables (Q : ℝ) (AB BC CA AD BE CF : ℝ)
noncomputable def Area (T : Type) [inner_product_space ℝ T] := sorry

noncomputable def problem_statement : Prop :=
  ∀ (A B C D E F : Type) [inner_product_space ℝ A] 
    [inner_product_space ℝ B] [inner_product_space ℝ C]
    [inner_product_space ℝ D] [inner_product_space ℝ E] 
    [inner_product_space ℝ F] (Q : ℝ) (3Q_area : Area (triangle ABC) = 3Q)
    (AD_EQ_AB : AD = 1/3 * AB) (BE_EQ_BC : BE = 1/3 * BC) (CF_EQ_CA : CF = 1/3 * CA),
  Area (triangle DEF) = 2Q

theorem lean_proof : problem_statement :=
begin
  sorry
end

end lean_proof_l239_239912


namespace dividend_div_3_quot_16_rem_4_eq_52_l239_239209

theorem dividend_div_3_quot_16_rem_4_eq_52 :
  ∃ (dividend : ℕ), ∀ (divisor quotient remainder : ℕ),
  divisor = 3 → quotient = 16 → remainder = 4 → dividend = divisor * quotient + remainder → dividend = 52 :=
begin
  sorry
end

end dividend_div_3_quot_16_rem_4_eq_52_l239_239209


namespace part1_part2_l239_239603

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end part1_part2_l239_239603


namespace min_value_of_a_l239_239426

theorem min_value_of_a (a : ℝ) (a_sequence : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a_sequence 1 = 1 / 5)
  (h2 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a_sequence (n + m) = a_sequence n * a_sequence m)
  (h3 : ∀ n : ℕ, n > 0 → S n = ∑ k in range n, a_sequence (k + 1))
  (h4 : ∀ n : ℕ, n > 0 → S n < a)
  : a ≥ 1 / 4 := 
sorry

end min_value_of_a_l239_239426


namespace solution_set_for_f_l239_239643

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x - 4 else 2^(-x) - 4

theorem solution_set_for_f :
  (is_even_function f) →
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end solution_set_for_f_l239_239643


namespace max_marked_vertices_no_rectangle_l239_239732

-- Definitions for the conditions
def regular_polygon (n : ℕ) := n ≥ 3

def no_four_marked_vertices_form_rectangle (n : ℕ) (marked_vertices : Finset ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : ℕ), 
  v1 ∈ marked_vertices ∧ 
  v2 ∈ marked_vertices ∧ 
  v3 ∈ marked_vertices ∧ 
  v4 ∈ marked_vertices → 
  (v1, v2, v3, v4) do not form the vertices of a rectangle in a regular n-gon

-- The theorem we need to prove
theorem max_marked_vertices_no_rectangle (marked_vertices : Finset ℕ) :
  marked_vertices.card ≤ 1009 :=
begin
  assume h1: regular_polygon 2016,
  assume h2: no_four_marked_vertices_form_rectangle 2016 marked_vertices,
  sorry
end

end max_marked_vertices_no_rectangle_l239_239732


namespace product_of_odd_primes_less_32_mod_32_l239_239185

open Nat

/-- The product of all odd primes less than 2^5 gives a remainder of 21 when divided by 32. -/
theorem product_of_odd_primes_less_32_mod_32 :
  let P := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
  in P % 32 = 21 := by
  let P := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
  have : 2^5 = 32 := by rfl
  exact sorry

end product_of_odd_primes_less_32_mod_32_l239_239185


namespace problem1_problem2_l239_239526

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239526


namespace matrix_projection_2_1_3_4_l239_239631

noncomputable def projection_matrix (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ w, let u := v.1 * w.1 + v.2 * w.2 in
  let norm_sq := v.1 * v.1 + v.2 * v.2 in
  (u / norm_sq * v.1, u / norm_sq * v.2)

def matrix_of_projection (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) : (ℝ × ℝ → ℝ × ℝ) :=
  λ v0, let v1_proj := projection_matrix v1 v0 in
  projection_matrix v2 v1_proj

theorem matrix_projection_2_1_3_4 (v0 : ℝ × ℝ) :
  (matrix_of_projection (2, -1) (3, 4) v0) =
  ( ( (-6/125 : ℝ) * v0.1 + (3/125 : ℝ) * v0.2 ),
    ( (-8/125 : ℝ) * v0.1 + (4/125 : ℝ) * v0.2 ) ) := sorry

end matrix_projection_2_1_3_4_l239_239631


namespace second_machine_time_equation_l239_239796

-- Define the rates and conditions
def first_machine_rate : ℝ := 1000 / 12
def combined_rate : ℝ := 1000 / 4

-- Proving the equation for the second machine time
theorem second_machine_time_equation (x : ℝ) : 
  first_machine_rate + (1000 / x) = combined_rate → (1 / 12 + 1 / x = 1 / 4) :=
by
  sorry

end second_machine_time_equation_l239_239796


namespace barbara_shopping_l239_239349

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l239_239349


namespace estimate_sqrt_diff_l239_239017

-- Defining approximate values for square roots
def approx_sqrt_90 : ℝ := 9.5
def approx_sqrt_88 : ℝ := 9.4

-- Main statement
theorem estimate_sqrt_diff : |(approx_sqrt_90 - approx_sqrt_88) - 0.10| < 0.01 := by
  sorry

end estimate_sqrt_diff_l239_239017


namespace count_integers_containing_zero_l239_239576

def has_digit_zero (n : ℕ) : Prop :=
  (nat.digits 10 n).contains 0

theorem count_integers_containing_zero :
  (Finset.filter has_digit_zero (Finset.range (3017 + 1))).card = 1011 := sorry

end count_integers_containing_zero_l239_239576


namespace solution_set_of_inequality_l239_239133

variable {R : Type} [OrderedRing R] 

def is_even (f : R → R) := ∀ x : R, f (-x) = f x
def is_increasing_on (f : R → R) (s : set R) := ∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_even : is_even f) (h_incr : is_increasing_on f (set.Iic 0)) :
  {x : ℝ | f (x - 1) ≥ f 1} = set.Icc 0 2 :=
begin
  sorry
end

end solution_set_of_inequality_l239_239133


namespace part1_solution_set_part2_range_a_l239_239555

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239555


namespace M_inter_N_l239_239926

def M : Set ℝ := { x | -2 < x ∧ x < 1 }
def N : Set ℤ := { x | Int.natAbs x ≤ 2 }

theorem M_inter_N : { x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) < 1 } ∩ N = { -1, 0 } :=
by
  simp [M, N]
  sorry

end M_inter_N_l239_239926


namespace cost_to_paint_cube_is_16_l239_239584

def edge_length (cube : ℝ) : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def one_quart_coverage : ℝ := 120
def num_faces : ℕ := 6

theorem cost_to_paint_cube_is_16 : 
  let area_of_one_face := (edge_length 10) ^ 2 in
  let total_surface_area := num_faces * area_of_one_face in
  let quarts_needed := total_surface_area / one_quart_coverage in
  let total_cost := quarts_needed * paint_cost_per_quart in
  total_cost = 16 := 
by
  sorry

end cost_to_paint_cube_is_16_l239_239584


namespace problem_solution_solution_is_frac_l239_239170

theorem problem_solution (x : ℚ) (hx: x.floor = 3) :
  x * ((x.floor : ℚ) * ((x.floor : ℚ) * (x.floor : ℚ) * x).floor : ℚ).floor = 88 := 
begin
  sorry,
end

theorem solution_is_frac (q : ℚ) : q = 22 / 7 :=
begin
  sorry,
end

end problem_solution_solution_is_frac_l239_239170


namespace distance_sum_l239_239230

variables {A B C D P M I : Type} [euclidean_geometry A B C D P M I]

def angle_bisector_intersects_circumcircle (A B C D : Type) [euclidean_geometry A B C D] : Prop :=
  ∃ (D : Point),
  (angle_bisector A B C D)

def symmetric_point (P I M : Type) [euclidean_geometry P I M] : Prop :=
  ∃ (P : Point), 
  is_reflection P I M

def second_intersection_point (D P M : Type) [euclidean_geometry D P M] : Prop :=
  ∃ (M : Point), 
  intersects_circumcircle (line_through D P) M

theorem distance_sum (A B C D P M I : Type) [euclidean_geometry A B C D P M I]
  (x : angle_bisector_intersects_circumcircle A B C D)
  (y : symmetric_point P I M)
  (z : second_intersection_point D P M) :
  distance M A = distance M B + distance M C :=
sorry

end distance_sum_l239_239230


namespace _l239_239505

noncomputable def equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : 0 < b) (h3 : a = 2) (h4 : a*b = 1*sqrt(3)) : 
  Prop := (x^2 / 4 + y^2 / 3 = 1)

noncomputable theorem ellipse_property (a b : ℝ) 
  (h1 : a > b) (h2 : 0 < b) (h3 : a = 2) (h4 : a = sqrt(4)) 
  (h5 : ∃ k m : ℝ, let l := (y = kx + m) in ∃ A B : ℝ × ℝ, A * B = sorry) 
  (h6 : ∃ Q : ℝ × ℝ, Q = (-4, m - 4*k)) (h7 : ∃ P : ℝ × ℝ, P ∈ ellipse(E) ∧
  P = A + B ∧ P • Q = 3/2) :
  equation_of_ellipse a b h1 h2 h3 h4 := 
sorry

end _l239_239505


namespace expected_value_of_three_marbles_l239_239120

-- Define the set of marbles
def marbles := {1, 2, 3, 4, 5, 6}

-- Define the set of possible combinations of drawing 3 marbles
def combinations := marbles.powerset.filter (λ s, s.card = 3)

-- Define the sum of the elements in a set
def sum_set (s : Finset ℕ) : ℕ := s.sum id

-- Define the expected value of the sum of the numbers on the drawn marbles
def expected_value : ℚ :=
  (Finset.sum combinations sum_set : ℚ) / combinations.card

theorem expected_value_of_three_marbles :
  expected_value = 10.05 := sorry

end expected_value_of_three_marbles_l239_239120


namespace trajectory_of_P_cosine_angle_EOF_values_of_a_l239_239087

section trajectory_proof

variable (θ a : ℝ)
variable (E F O P A M : ℝ × ℝ)

-- Fixed point A
def A : ℝ × ℝ := (12, 0)

-- Curve for M
def curve_M (θ : ℝ) : ℝ × ℝ :=
  (6 + 2 * Real.cos θ, 2 * Real.sin θ)

-- AP vector
def AP (P : ℝ × ℝ) : (ℝ × ℝ) := (P.1 - A.1, P.2)

-- AM vector
def AM (θ : ℝ) : (ℝ × ℝ) := let M := curve_M θ in (M.1 - A.1, M.2)

-- Condition: AP = 2 * AM
axiom AP_twice_AM (P : ℝ × ℝ) (θ : ℝ) : AP P = (2 * AM θ).1

/-
Prove:
1. The equation of the trajectory C of the moving point P is (x - 12)^2 + y^2 = 16.
-/
theorem trajectory_of_P (θ : ℝ) (P : ℝ × ℝ) :
  ∀ P, P = (4 * Real.cos θ + 12, 4 * Real.sin θ) → (P.1 - 12)^2 + P.2^2 = 16 := by
  sorry

-- Line l 
def line_l (a : ℝ) : ℝ × ℝ → Prop := fun P => P.2 = -P.1 + a

-- Points E and F on the curve that intersects with line l
def line_intersects_curve_C (E F : ℝ × ℝ) (a : ℝ) : Prop :=
  (line_l a E) ∧ (line_l a F) ∧ ((E - (12,0)).fst^2 + E.snd^2 = 16) ∧ ((F - (12,0)).fst^2 + F.snd^2 = 16)

-- scalar product of OE and OF
def vector_dot_product (OE OF : ℝ × ℝ) : ℝ :=
  OE.1 * OF.1 + OE.2 * OF.2

-- intersection condition
axiom intersection_condition (E F : ℝ × ℝ) : vector_dot_product E F = 12

/-
2. The cosine of ∠EOF is 3/4.
-/
theorem cosine_angle_EOF (E F O : ℝ × ℝ) :
  (∠ EOF) = (3/4) := by
  sorry

/-
3. The values of a are ±2√7.
-/
theorem values_of_a (a : ℝ) :
  a = 2*Real.sqrt 7 ∨ a = -2*Real.sqrt 7 := by
  sorry

end trajectory_proof

end trajectory_of_P_cosine_angle_EOF_values_of_a_l239_239087


namespace range_of_a_for_monotonic_f_l239_239175

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + 5 * x + 6

theorem range_of_a_for_monotonic_f :
  (∀ x y ∈ set.Icc (1:ℝ) 3, f a x ≤ f a y ∨ f a y ≤ f a x)
  ↔ (a ∈ set.Icc (-(real.sqrt 5)) ∞ ∪ set.Icc (-∞) (-3 : ℝ)) :=
sorry

end range_of_a_for_monotonic_f_l239_239175


namespace lcm_of_times_l239_239254

-- Define the times each athlete takes to complete one lap
def time_A : Nat := 4
def time_B : Nat := 5
def time_C : Nat := 6

-- Prove that the LCM of 4, 5, and 6 is 60
theorem lcm_of_times : Nat.lcm time_A (Nat.lcm time_B time_C) = 60 := by
  sorry

end lcm_of_times_l239_239254


namespace find_linear_function_l239_239088

theorem find_linear_function (a : ℝ) (a_pos : 0 < a) :
  ∃ (b : ℝ), ∀ (f : ℕ → ℝ),
  (∀ (k m : ℕ), (a * m ≤ k ∧ k < (a + 1) * m) → f (k + m) = f k + f m) →
  ∀ n : ℕ, f n = b * n :=
sorry

end find_linear_function_l239_239088


namespace add_numerator_denominator_add_numerator_denominator_gt_one_l239_239656

variable {a b n : ℕ}

/-- Adding the same natural number to both the numerator and the denominator of a fraction 
    increases the fraction if it is less than one, and decreases the fraction if it is greater than one. -/
theorem add_numerator_denominator (h1: a < b) : (a + n) / (b + n) > a / b := sorry

theorem add_numerator_denominator_gt_one (h2: a > b) : (a + n) / (b + n) < a / b := sorry

end add_numerator_denominator_add_numerator_denominator_gt_one_l239_239656


namespace geometric_sequence_properties_l239_239790

theorem geometric_sequence_properties (n : ℕ) (hn : 0 < n) :
  let a_1 := 1 / 2
  let q := 1 / 2
  let a_n := a_1 * q^(n - 1)
  let S_n := a_1 * (1 - q^n) / (1 - q)
  in a_1 > 0 ∧ (S_n < 1) ∧ (a_n = 1 / 2^n) :=
by {
  let a_1 := 1 / 2;
  let q := 1 / 2;
  let a_n := a_1 * q^(n - 1);
  let S_n := a_1 * (1 - q^n) / (1 - q);
  have h1 : a_1 > 0 := by norm_num;
  have h2 : S_n < 1 := sorry;
  have h3 : a_n = 1 / 2^n := sorry;
  exact ⟨h1, h2, h3⟩;
}

end geometric_sequence_properties_l239_239790


namespace unicorn_rope_problem_l239_239331

noncomputable def turret_rope_contact_length (a b c : ℕ) := 
  a - real.sqrt b / c

theorem unicorn_rope_problem
  (turret_radius : ℝ) (rope_length : ℝ) (unicorn_height : ℝ) (horizontal_distance : ℝ)
  (a b c : ℕ) : 
  turret_radius = 10 ∧
  rope_length = 30 ∧
  unicorn_height = 5 ∧
  horizontal_distance = 5 ∧
  prime c ∧
  turret_rope_contact_length a b c = 30 - 5*real.sqrt(5)
  → a + b + c = 843 :=
by sorry

end unicorn_rope_problem_l239_239331


namespace smallest_solution_l239_239063

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239063


namespace smallest_solution_exists_l239_239051

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239051


namespace f_perf_square_iff_n_eq_one_l239_239069

def f (n : ℕ+) : ℕ :=
  (Finset.filter (fun s : Finset (Fin n) => nat.gcd (s.val.filter (λ x, x ∈ s)) = 1)
  (Finset.powerset (Finset.range (n : ℕ)))).card

theorem f_perf_square_iff_n_eq_one (n : ℕ+) : (∃ k : ℕ, k * k = f n) ↔ n = 1 := sorry

end f_perf_square_iff_n_eq_one_l239_239069


namespace units_digit_six_l239_239280

theorem units_digit_six (n : ℕ) (h : n > 0) : (6 ^ n) % 10 = 6 :=
by sorry

example : (6 ^ 7) % 10 = 6 :=
units_digit_six 7 (by norm_num)

end units_digit_six_l239_239280


namespace prove_R36_div_R6_minus_R3_l239_239371

noncomputable def R (k : ℕ) : ℤ := (10^k - 1) / 9

theorem prove_R36_div_R6_minus_R3 :
  (R 36 / R 6) - R 3 = 100000100000100000100000100000099989 := sorry

end prove_R36_div_R6_minus_R3_l239_239371


namespace inequality_solution_l239_239666

theorem inequality_solution : 
  ∀ x : ℝ, 
  (2*x^2 + 7*x + 3 ≥ 0) →
  (3*x + 4 - 2*real.sqrt(2*x^2 + 7*x + 3)) * (abs(x^2 - 4*x + 2) - abs(x - 2)) ≤ 0 ↔ 
  x ∈ set.Iic (-3) ∪ set.Icc 0 1 ∪ {2} ∪ set.Icc 3 4 :=
by
  intro x,
  intro h_domain,
  sorry

end inequality_solution_l239_239666


namespace ABCD_is_parallelogram_l239_239235

theorem ABCD_is_parallelogram
  (S1 S2 : Set Point) (A P B C D : Point)
  (h1 : A ∈ S1 ∧ A ∈ S2)
  (h2 : P ∈ S1 ∧ P ∈ S2)
  (h3 : tangent AB S1 A)  -- Placeholder for tangent definition
  (h4 : is_parallel AB CD)
  (h5 : passes_through CD P)
  (h6 : B ∈ S2)
  (h7 : C ∈ S2)
  (h8 : D ∈ S1) :
  parallelogram ABCD := 
sorry

end ABCD_is_parallelogram_l239_239235


namespace hyperbola_focal_length_l239_239491

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239491


namespace smallest_solution_l239_239057

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239057


namespace hyperbola_focal_length_is_4_l239_239450

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239450


namespace common_point_of_arithmetic_progression_lines_l239_239840

theorem common_point_of_arithmetic_progression_lines 
  (a d : ℝ) 
  (h₁ : a ≠ 0)
  (h_d_ne_zero : d ≠ 0) 
  (h₃ : ∀ (x y : ℝ), (x = -1 ∧ y = 1) ↔ (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = (a-2*d))) :
  (∀ (x y : ℝ), (a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = a-2*d) → x = -1 ∧ y = 1) :=
by 
  sorry

end common_point_of_arithmetic_progression_lines_l239_239840


namespace xiao_hong_additional_time_l239_239759

/-- Xiao Hong needs 2 more minutes to reach the newsstand than Xiao Hua,
    given Xiao Hua's speed is 70 meters per minute, Xiao Hong's speed is 60 meters per minute,
    and the time for Xiao Hua to reach the newsstand is 12 minutes. -/
theorem xiao_hong_additional_time :
  ∀ (speed_hua speed_hong time_hua : ℕ),
  speed_hua = 70 → speed_hong = 60 → time_hua = 12 →
  let distance := speed_hua * time_hua in
  let time_hong := distance / speed_hong in
  (time_hong - time_hua) = 2 :=
by
  intros speed_hua speed_hong time_hua h1 h2 h3
  simp [h1, h2, h3]
  let distance := 70 * 12
  let time_hong := distance / 60
  have h_time_hong : time_hong = 14 := by norm_num
  rw h_time_hong
  norm_num

end xiao_hong_additional_time_l239_239759


namespace percent_non_sugar_l239_239325

-- Definitions based on the conditions in the problem.
def pie_weight : ℕ := 200
def sugar_weight : ℕ := 50

-- Statement of the proof problem.
theorem percent_non_sugar : ((pie_weight - sugar_weight) * 100) / pie_weight = 75 :=
by
  sorry

end percent_non_sugar_l239_239325


namespace part1_solution_part2_solution_l239_239548

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239548


namespace consecutive_sum_is_10_l239_239710

theorem consecutive_sum_is_10 (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) : a + 2 = 10 :=
sorry

end consecutive_sum_is_10_l239_239710


namespace sequence_property_l239_239637

variable {Z : Type} [Int Z]

theorem sequence_property (m : Z) (h1 : abs m ≥ 2)
  (a : ℕ → Z) (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n, a (n + 2) = a (n + 1) - m * a n)
  (r s : ℤ) (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ abs m := sorry

end sequence_property_l239_239637


namespace polynomial_evaluation_l239_239623

theorem polynomial_evaluation (P Q : ℝ → ℝ) (r : ℝ) (hP : P = λ x, x^3 - 2 * x + 1)
  (hQ : Q = λ x, x^3 - 4 * x^2 + 4 * x - 1) (hPr : P r = 0) : Q (r^2) = 0 := 
sorry

end polynomial_evaluation_l239_239623


namespace crackers_eaten_l239_239649

-- Define the number of packs and their respective number of crackers
def num_packs_8 : ℕ := 5
def num_packs_10 : ℕ := 10
def num_packs_12 : ℕ := 7
def num_packs_15 : ℕ := 3

def crackers_per_pack_8 : ℕ := 8
def crackers_per_pack_10 : ℕ := 10
def crackers_per_pack_12 : ℕ := 12
def crackers_per_pack_15 : ℕ := 15

-- Calculate the total number of animal crackers
def total_crackers : ℕ :=
  (num_packs_8 * crackers_per_pack_8) +
  (num_packs_10 * crackers_per_pack_10) +
  (num_packs_12 * crackers_per_pack_12) +
  (num_packs_15 * crackers_per_pack_15)

-- Define the number of students who didn't eat their crackers and the respective number of crackers per pack
def num_students_not_eaten : ℕ := 4
def different_crackers_not_eaten : List ℕ := [8, 10, 12, 15]

-- Calculate the total number of crackers not eaten by adding those packs.
def total_crackers_not_eaten : ℕ := different_crackers_not_eaten.sum

-- Theorem to prove the total number of crackers eaten.
theorem crackers_eaten : total_crackers - total_crackers_not_eaten = 224 :=
by
  -- Total crackers: 269
  -- Subtract crackers not eaten: 8 + 10 + 12 + 15 = 45
  -- Therefore: 269 - 45 = 224
  sorry

end crackers_eaten_l239_239649


namespace ab_minus_2b2_eq_neg6_minus_3i_l239_239264

-- Define a and b as complex numbers.
def a : Complex := 1 + 2 * Complex.i
def b : Complex := 2 + Complex.i

-- State the theorem using the conditions and the correct answer.
theorem ab_minus_2b2_eq_neg6_minus_3i : a * b - 2 * b^2 = -6 - 3 * Complex.i :=
by
  -- Skipping the proof.
  sorry

end ab_minus_2b2_eq_neg6_minus_3i_l239_239264


namespace club_student_inequality_l239_239192

theorem club_student_inequality (n m : ℕ) (a : Fin n → ℕ) (h : ∀ i j : Fin n, i ≠ j → ∃ c : Fin m, a i ∈ c ∧ a j ∉ c) :
  (Finset.univ.sum (λ i : Fin n, Nat.factorial (a i) * Nat.factorial (m - a i))) ≤ Nat.factorial m :=
sorry

end club_student_inequality_l239_239192


namespace div_scalar_vector_l239_239660

variables {α : Type*} [RealField α]
variables (u : α → α) (a : α → α → α)

theorem div_scalar_vector (u : α → α) (a : α → α → α) :
  div (λ x, u x * a x) = (λ x, u x * div a x) + (λ x, dot (a x) (grad u x)) :=
sorry

end div_scalar_vector_l239_239660


namespace collinear_C_E_B_l239_239370

noncomputable theory

variables {A B C D E O1 O2 : Point}
variables (circle1 circle2 : Circle)
variables (l : Line)

-- Define conditions
def tangent_to_line_at_E (circle : Circle) (l : Line) (E : Point) : Prop :=
  circle.tangent_point l E

def parallel_diameters (AB DC : Line) : Prop :=
  AB.parallel DC

def collinear_points (P Q R : Point) : Prop :=
  Line.through P Q = Line.through P R

-- Known properties
axiom tangent_pt : tangent_to_line_at_E circle1 l E
axiom tangent_pt' : tangent_to_line_at_E circle2 l E
axiom diam_parallel : parallel_diameters (Diameter circle1 A B) (Diameter circle2 D C)
axiom collinear_O1_O2_E : collinear_points O1 O2 E
axiom equal_angles : ∠ B O1 E = ∠ C O2 E
axiom radii_equal : (distance O1 E) = (distance O2 E)

-- Theorem to prove collinearity
theorem collinear_C_E_B (h1 : tangent_to_line_at_E circle1 l E)
                         (h2 : tangent_to_line_at_E circle2 l E)
                         (h3 : parallel_diameters (Diameter circle1 A B) (Diameter circle2 D C))
                         (h4 : collinear_points O1 O2 E)
                         (h5 : ∠ B O1 E = ∠ C O2 E)
                         (h6 : (distance O1 E) = (distance O2 E)) :
  collinear_points C E B :=
  sorry

end collinear_C_E_B_l239_239370


namespace octahedron_in_cube_l239_239422

theorem octahedron_in_cube :
  ∀ (C : Type) [cube C],
  (∀ (P : convex_polyhedron C), (∃ v : C, v ∈ edges(C) ∧ ∀ e ∈ edges(C), ∃! v ∈ P)) →
  (∃ O : convex_polyhedron C, 
    (∀ v ∈ vertices(O), ∃ f ∈ faces(C), v = center(f)) ∧ 
    ∀ x, (x ∈ O ↔ (∀ P, (∃ v : C, v ∈ edges(C) ∧ ∀ e ∈ edges(C), ∃! v ∈ P) → x ∈ P))) :=
sorry

end octahedron_in_cube_l239_239422


namespace smallest_solution_l239_239042

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239042


namespace number_of_common_tangents_l239_239891

-- Definitions based on the given conditions
def circle_M := { p : ℝ × ℝ | p.1 + p.2 ^ 2 = 1 }
def circle_N := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 * p.2 - 11 = 0 }

-- The theorem we want to prove
theorem number_of_common_tangents : 
  -- Statement that the number of common tangents between circle_M and circle_N is 1
  ∃ n : ℕ, n = 1 ∧ (∀ l : (ℝ × ℝ) → Prop, tangent l circle_M ∧ tangent l circle_N → n = 1) :=
begin
  sorry
end

end number_of_common_tangents_l239_239891


namespace triangle_problem_l239_239157

noncomputable def vector_magnitude (x y z : ℝ) : ℝ := 
  real.sqrt (x^2 + y^2 - 2 * x * y * z)

theorem triangle_problem (AC AB : ℝ) (hAC : AC = 1) (hAB : AB = 2) (cosA : ℝ) (hcosA : cosA = 1 / 8)
  (μ : ℝ) (hBD : (λ λ : ℝ, λ = 2) = λ → 2 * AC + μ * AB) : 
  vector_magnitude (2 * AC) AB (1 / 8) = real.sqrt 7 := 
by 
  sorry

end triangle_problem_l239_239157


namespace sandy_correct_sums_l239_239659

variable (c i : ℕ)

theorem sandy_correct_sums (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by {
  -- Proof goes here
  sorry
}

end sandy_correct_sums_l239_239659


namespace circumcircles_are_tangent_l239_239814

-- In Lean, we represent points, triangles, circles, etc., as types and their relationships.
variables {Point : Type} (A B C H M N P Q S T U V X Y Z : Point)
variables {Triangle : Type} (ABC XYZ : Triangle) {Circle : Type} (Γ : Circle)

-- Definitions to describe the geometry and points
def isAcuteAngledTriangleInscribedInCircle (Δ : Triangle) (C : Circle) : Prop := sorry
def isOrthocenterOf (H : Point) (Δ : Triangle) : Prop := sorry
def isMidpointOf (M : Point) (p1 p2 : Point) : Prop := sorry
def intersectsAt (l : Line) (C : Circle) (p : Point) : Prop := sorry
def symmetricTo (p q : Point) (m : Point) : Prop := sorry
def circumcircle (Δ : Triangle) : Circle := sorry
def areTangent (C1 C2 : Circle) : Prop := sorry

-- Main theorem statement
theorem circumcircles_are_tangent 
  (h1 : isAcuteAngledTriangleInscribedInCircle ABC Γ)
  (h2 : isOrthocenterOf H ABC)
  (h3 : isMidpointOf M B C)  -- Assuming B, C are sides for simplicity
  (h4 : intersectsAt (Line.mk N Γ) Γ N)
  (h5 : P ∉ H)  -- Point P on the arc not containing H
  (h6 : symmetricTo S P U)
  (h7 : symmetricTo T P V) :
  areTangent (circumcircle XYZ) (circumcircle ABC) :=
sorry

end circumcircles_are_tangent_l239_239814


namespace units_digit_of_product_of_odd_numbers_not_ending_in_5_l239_239247

theorem units_digit_of_product_of_odd_numbers_not_ending_in_5 :
  let N := (List.filter (λ n, n % 2 = 1 ∧ n % 10 ≠ 5) (List.range' 1 100)).prod
  Nat.units_digit N = 1 :=
by
  sorry

end units_digit_of_product_of_odd_numbers_not_ending_in_5_l239_239247


namespace minimum_bills_and_coins_l239_239219

-- Definitions of the denominations involved.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def one_dollar := 100
def two_dollar := 200

-- The total change required.
def change := 456

-- Prove that the minimal combination of items to make $4.56 in change is 6.
theorem minimum_bills_and_coins : 
  ∃ (b2 : ℕ) (q : ℕ) (n : ℕ) (p : ℕ), 
  b2 * two_dollar + q * quarter + n * nickel + p * penny = change ∧ 
  b2 + q + n + p = 6 := 
by 
  exists 2, 2, 1, 1
  split
  { -- Check the total change
    calc 
      2 * two_dollar + 2 * quarter + 1 * nickel + 1 * penny
            = 2 * 200 + 2 * 25 + 1 * 5 + 1 * 1 := by rfl
        ... = 400 + 50 + 5 + 1 := by rfl
        ... = change := by rfl },
  { -- Total items count
    exact rfl }

end minimum_bills_and_coins_l239_239219


namespace smallest_solution_to_equation_l239_239030

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239030


namespace sum_of_angles_is_3pi_over_4_l239_239910

variable {α β : ℝ} -- variables representing angles

-- conditions
axiom acute_angles (hα : α < π / 2) (hβ : β < π / 2)
axiom tan_roots (h₁ : ∃ (x y : ℝ), (x^2 - 5 * x + 6 = 0) ∧ (y^2 - 5 * y + 6 = 0) ∧ (tan α = x ∨ tan α = y) ∧ (tan β = x ∨ tan β = y) ∧ x ≠ y)

-- proposition to prove
theorem sum_of_angles_is_3pi_over_4 (hα : α < π / 2) (hβ : β < π / 2)
  (h_tan_roots : ∃ (x y : ℝ), (x^2 - 5 * x + 6 = 0) ∧ (y^2 - 5 * y + 6 = 0) 
    ∧ (tan α = x ∨ tan α = y) ∧ (tan β = x ∨ tan β = y) ∧ x ≠ y) :
  α + β = 3 * π / 4 := 
sorry

end sum_of_angles_is_3pi_over_4_l239_239910


namespace part1_solution_set_part2_range_a_l239_239558

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239558


namespace hyperbola_focal_length_l239_239479

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239479


namespace find_speed_v_l239_239780

-- Definitions of the given constants
def speed1 := 65.45454545454545
def time_diff := 10
def distance := 1

-- Define the unknown speed
def speed2 (v : ℝ) : Prop :=
  let time1 := distance / (speed1 / 3600)
  let time2 := distance / (v / 3600)
  time1 - time2 = time_diff

-- The theorem we want to prove
theorem find_speed_v : speed2 80 :=
  by
  -- Adding placeholder proof, as required
  sorry

end find_speed_v_l239_239780


namespace term_x4_in_expansion_l239_239188

-- Define the imaginary unit.
def i : ℂ := Complex.I

-- Define the binomial coefficient.
def binom : ℕ → ℕ → ℕ 
| n k := Nat.choose n k

-- Define the term we are interested in based on the binomial expansion formula.
def term_in_binomial_expansion (n k : ℕ) (a b : ℂ) : ℂ :=
  binom n k * (a^(n-k)) * (b^k)

-- Define the theorem stating the answer of the problem.
theorem term_x4_in_expansion :
  term_in_binomial_expansion 6 4 (x: ℂ) i = -15 * x^4 :=
by
  sorry

end term_x4_in_expansion_l239_239188


namespace quadratic_rational_coeff_l239_239386

theorem quadratic_rational_coeff (x : ℤ) : 
  (∃ (a b c : ℚ), a = 1 ∧ b = 6 ∧ c = 14 ∧ 
  (sqrt 5 - 3) ∈ {r | a * r^2 + b * r + c = 0}) :=
by
  use [1, 6, 14]
  split; norm_num
  sorry

end quadratic_rational_coeff_l239_239386


namespace ratio_problem_l239_239123

theorem ratio_problem
  (a b c d e : ℚ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2) :
  e / a = 2 / 35 := 
sorry

end ratio_problem_l239_239123


namespace part1_solution_part2_solution_l239_239550

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239550


namespace min_coins_for_any_amount_below_dollar_l239_239728

-- Definitions of coin values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- Statement: The minimum number of coins required to pay any amount less than a dollar
theorem min_coins_for_any_amount_below_dollar :
  ∃ (n : ℕ), n = 11 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount < 100 →
   ∃ (a b c d : ℕ), amount = a * penny + b * nickel + c * dime + d * half_dollar ∧ 
   a + b + c + d ≤ n) :=
sorry

end min_coins_for_any_amount_below_dollar_l239_239728


namespace geometric_mean_equality_l239_239099

theorem geometric_mean_equality
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h : (a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18 + a 19 + a 20) / 10 = 
       (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + 
        a 13 + a 14 + a 15 + a 16 + a 17 + a 18 + a 19 + a 20 + a 21 + a 22 + 
        a 23 + a 24 + a 25 + a 26 + a 27 + a 28 + a 29 + a 30) / 30) :
  ( ∏ i in finset.range 10 \ finset.range 10, b (11 + i))^(1 / 10) = 
    ( ∏ i in finset.range 30 \ finset.range 30, b (1 + i))^(1 / 30) :=
sorry

end geometric_mean_equality_l239_239099


namespace monotonically_increasing_on_pos_real_l239_239285

def f1 (x : ℝ) := - real.log x
def f2 (x : ℝ) := 1 / 2^x
def f3 (x : ℝ) := -1 / x
def f4 (x : ℝ) := 3^|x-1|

theorem monotonically_increasing_on_pos_real (f : ℝ → ℝ)
    (h1 : ∀ x, f = f1 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h2 : ∀ x, f = f2 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h3 : ∀ x, f = f3 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h4 : ∀ x, f = f4 → x ∈ set.Ioi 0 →  deriv f x > 0) :
  f = f3 → ∀ x ∈ set.Ioi 0, deriv f x > 0 := 
sorry

end monotonically_increasing_on_pos_real_l239_239285


namespace imag_part_z_zero_l239_239919

-- Define the complex number z
noncomputable def z : ℂ :=
  let i : ℂ := complex.I in
  (2 : ℂ) / ((i - 1) * (i + 1))

-- Define the imaginary part
noncomputable def imag_part_z : ℂ := complex.im(z)

-- Lean 4 Theorem Statement to prove the imaginary part is 0
theorem imag_part_z_zero : imag_part_z = 0 :=
by
  -- Proof will be added here
  sorry

end imag_part_z_zero_l239_239919


namespace probability_Alice_three_turns_l239_239333

-- Define probabilities
def P_AliceKeeps (p_AliceToAlice: ℚ) : ℚ := 2 / 3
def P_AliceToBob (p_AliceToBob: ℚ) : ℚ := 1 / 3
def P_BobToAlice (p_BobToAlice: ℚ) : ℚ := 1 / 4
def P_BobKeeps (p_BobKeeps: ℚ) : ℚ := 3 / 4

-- Calculate the probability for each path
def P_AAAA : ℚ := P_AliceKeeps () * P_AliceKeeps () * P_AliceKeeps ()
def P_AABA : ℚ := P_AliceKeeps () * P_AliceKeeps () * P_BobToAlice ()
def P_ABAA : ℚ := P_AliceToBob () * P_BobToAlice () * P_AliceKeeps ()
def P_ABBA : ℚ := P_AliceToBob () * P_BobKeeps () * P_BobToAlice ()

-- Total probability calculation
def total_probability : ℚ :=
  P_AAAA + P_AABA + P_ABAA + P_ABBA

-- The proof statement
theorem probability_Alice_three_turns :
  total_probability = 227 / 432 := by
  sorry

end probability_Alice_three_turns_l239_239333


namespace abs_inequality_solution_l239_239066

theorem abs_inequality_solution (x : ℝ) : 2 * |x - 1| - 1 < 0 ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end abs_inequality_solution_l239_239066


namespace correct_statement_among_conditions_l239_239338

theorem correct_statement_among_conditions :
  (∀ (α β : ℝ), α > β → ¬ (sin α > sin β)) ∧
  (¬ (∀ (x : ℝ), x > 1 → x^2 > 1) ↔ (∃ (x : ℝ), x > 1 ∧ x^2 ≤ 1)) ∧
  (∀ (x : ℝ), (x ≤ 4 / 3) → (1 / (x - 1) ≥ 3)) ∧
  (∀ (x : ℝ), (1 / (x - 1) ≥ 3) → (x ≤ 4 / 3)) ∧
  (∀ (x y : ℝ), (x * y = 0 → (x = 0 ∨ y = 0))) ∧
  (∀ (x y : ℝ), ((x ≠ 0) ∧ (y ≠ 0)) → (x * y ≠ 0)) →
  "C" = "C"
:= sorry

end correct_statement_among_conditions_l239_239338


namespace problem1_problem2_l239_239520

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239520


namespace sequence_general_term_and_sum_l239_239916

-- Definitions based on conditions
def arithmetic_progression (a : ℕ → ℕ) : Prop :=
  ∃ (a₁ d : ℕ), ∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n) / 2)

def sequence_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 9 = 24 ∧ (5 * (a 1 + a 5) / 2) = 30

-- The sequence and its sum satisfy the given conditions
theorem sequence_general_term_and_sum {a : ℕ → ℕ} {S : ℕ → ℕ} (T : ℕ → ℕ)
  (h1 : arithmetic_progression a)
  (h2 : sum_first_n_terms a S)
  (h3 : sequence_condition a) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = n * (3 * n + 5) / (16 * (n + 1) * (n + 2))) :=
begin
  sorry
end

end sequence_general_term_and_sum_l239_239916


namespace quadratic_zeros_difference_l239_239248

theorem quadratic_zeros_difference :
  ∃ a b c m n : ℝ, 
    (a ≠ 0) ∧
    (∀ (x : ℝ), a*x^2 + b*x + c = 4*(x - 3)^2 - 9) ∧
    (5*5*a + 5*b + c = 7) ∧
    (m > n) ∧
    (a*m^2 + b*m + c = 0) ∧
    (a*n^2 + b*n + c = 0) ∧
    ((m - n) = 3) :=
begin
  apply exists.intro 4,
  apply exists.intro (-24),
  apply exists.intro 27,
  apply exists.intro (9/2 : ℝ),
  apply exists.intro (3/2 : ℝ),
  split,
  -- Proof a ≠ 0
  { norm_num },
  split,
  -- Proof vertex form
  { intro x,
    field_simp,
    calc
      4 * x^2 - 4 * 6 * x + 4 * 9 - 9
      = 4 * (x - 3)^2 - 9 : by ring,
    },
  split,
  -- Proof it goes through (5, 7)
  { field_simp,
    calc
      4 * 25 - 24 * 5 + 27
      = 7 : by norm_num, },
  split,
  -- Proof m > n
  { norm_num },
  split,
  -- Proof m is a zero
  { field_simp,
    calc
      4 * (9 / 2) ^ 2 - 24 * (9 / 2) + 27
      = 0 : by norm_num },
  split,
  -- Proof n is a zero
  { field_simp,
    calc
      4 * (3 / 2) ^ 2 - 24 * (3 / 2) + 27
      = 0 : by norm_num },
  -- Proof m - n = 3
  { norm_num },
end

end quadratic_zeros_difference_l239_239248


namespace sequence_a_property_sequence_a_initial_sequence_a_recursive_sequence_a_explicit_sequence_sum_l239_239899

noncomputable def sequence_a (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / (2 * n + 1)

theorem sequence_a_property (n : ℕ) (hn : n ≠ 0) : 
  sequence_a n ≠ 0 :=
by
  simp [sequence_a]
  sorry

theorem sequence_a_initial : sequence_a 1 = 1 / 3 :=
by
  simp [sequence_a]
  norm_num

theorem sequence_a_recursive (n : ℕ) (hn : 2 ≤ n) : 
  sequence_a (n - 1) - sequence_a n = 2 * sequence_a (n - 1) * sequence_a n :=
by
  simp [sequence_a]
  sorry

theorem sequence_a_explicit (n : ℕ) : sequence_a n = 1 / (2 * n + 1) :=
by
  simp [sequence_a]
  sorry

theorem sequence_sum (n : ℕ) : 
  ∑ i in Finset.range (n + 1), sequence_a i * sequence_a (i + 1) = n / (6 * n + 9) :=
by
  sorry

end sequence_a_property_sequence_a_initial_sequence_a_recursive_sequence_a_explicit_sequence_sum_l239_239899


namespace problem_k_leq_pi_l239_239673

theorem problem_k_leq_pi (n k : ℕ) (a : Fin k → ℕ) 
  (h_above_one : ∀ i, 1 < a i)
  (h_sorted : ∀ i j, i < j → a i < a j)
  (h_bounded : ∀ i, a i ≤ n)
  (h_div_property : ∃ i, a i ∣ ∏ j in Finset.univ.filter (λ j, j ≠ i), a j) :
  k ≤ Nat.primePi n := sorry

end problem_k_leq_pi_l239_239673


namespace triangle_construction_l239_239616

-- Assume A, B are distinct points in ℝ^2 and line e is given.
variables (A B : ℝ × ℝ) (e : set (ℝ × ℝ))

-- Given condition a = 2b, this can be expressed in terms of distance.
def dist (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
def a (C : ℝ × ℝ) : ℝ := dist C B
def b (C : ℝ × ℝ) : ℝ := dist C A

theorem triangle_construction (hA_ne_B : A ≠ B)
  (h_line_contains_C : ∀ C, C ∈ e → dist C B = 2 * dist C A) :
  ∃ C1 C2 : ℝ × ℝ, C1 ∈ e ∧ dist C1 B = 2 * dist C1 A ∧ 
                    (C2 ∈ e ∧ dist C2 B = 2 * dist C2 A ∨ C1 = C2) :=
sorry

end triangle_construction_l239_239616


namespace number_of_symmetries_l239_239834

-- Definition related to the problem
def alternating_pattern : Prop := 
  ∃ (ℓ : line) (t : ℓ → bool), ∀ (p : ℓ), (if t p then is_triangle p else is_circle p) ∧
                             (if t (p + 1) then is_circle (p + 1) else is_triangle (p + 1))

def preserves_pattern (f : line → line) : Prop :=
  ∀ ℓ t, (alternating_pattern t) → (alternating_pattern (λ p, t (f p)))

-- The problem statement as a Lean theorem
theorem number_of_symmetries : 
  ∃ n, n = 2 ∧ ∀ (f : line → line), f ≠ id ∧ preserves_pattern f → is_translation f ∨ is_perpendicular_reflection f := sorry

end number_of_symmetries_l239_239834


namespace edge_length_proof_l239_239319

noncomputable def edge_length_of_cube 
  (base_length : ℝ) (base_width : ℝ) (rise_in_water : ℝ) 
  (V_cube : ℝ) : ℝ :=
if h : base_length ∨ base_width ∨ rise_in_water = V_cube then
  (V_cube^(1/3))
else
  (0)

theorem edge_length_proof : 
  ∃ (a : ℝ), let base_length := 20
  let base_width := 15
  let rise_in_water := 5.76
  let V_displaced := base_length * base_width * rise_in_water 
  (a^3 = V_displaced) ∧ (a = 12) :=
begin
  use 12,
  simp [base_length, base_width, rise_in_water, V_displaced],
  norm_num
end

end edge_length_proof_l239_239319


namespace inequality_solution_l239_239667

theorem inequality_solution : 
  ∀ x : ℝ, 
  (2*x^2 + 7*x + 3 ≥ 0) →
  (3*x + 4 - 2*real.sqrt(2*x^2 + 7*x + 3)) * (abs(x^2 - 4*x + 2) - abs(x - 2)) ≤ 0 ↔ 
  x ∈ set.Iic (-3) ∪ set.Icc 0 1 ∪ {2} ∪ set.Icc 3 4 :=
by
  intro x,
  intro h_domain,
  sorry

end inequality_solution_l239_239667


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239963

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239963


namespace correct_option_B_l239_239813

-- Define decimal representation of the numbers
def dec_13 : ℕ := 13
def dec_25 : ℕ := 25
def dec_11 : ℕ := 11
def dec_10 : ℕ := 10

-- Define binary representation of the numbers
def bin_1101 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 1*4 + 0*2 + 1*1 = 13
def bin_10110 : ℕ := 2^(4) + 2^(2) + 2^(1)  -- 1*16 + 0*8 + 1*4 + 1*2 + 0*1 = 22
def bin_1011 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 0*4 + 1*2 + 1*1 = 11
def bin_10 : ℕ := 2^(1)  -- 1*2 + 0*1 = 2

theorem correct_option_B : (dec_13 = bin_1101) := by
  -- Proof is skipped
  sorry

end correct_option_B_l239_239813


namespace neznaika_discrepancy_l239_239290

-- Definitions of the given conditions
def correct_kilograms_to_kilolunas (kg : ℝ) : ℝ := kg / 0.24
def neznaika_kilograms_to_kilolunas (kg : ℝ) : ℝ := (kg * 4) * 1.04

-- Define the precise value of 1 kiloluna in kilograms for clarity
def one_kiloluna_in_kg : ℝ := (1 / 4) * 0.96

-- Function calculating the discrepancy percentage
def discrepancy_percentage (kg : ℝ) : ℝ :=
  let correct_value := correct_kilograms_to_kilolunas kg
  let neznaika_value := neznaika_kilograms_to_kilolunas kg
  ((correct_value - neznaika_value).abs / correct_value) * 100

-- Statement of the theorem to be proven
theorem neznaika_discrepancy :
  discrepancy_percentage 1 = 0.16 := sorry

end neznaika_discrepancy_l239_239290


namespace area_triangle_NOI_is_5_l239_239137

noncomputable def triangle_area_NOI (P Q R : ℝ × ℝ) (PQ PR QR : ℝ) (O I N : ℝ × ℝ) : ℝ :=
  let A := P.1 * (I.2 - N.2) + I.1 * (N.2 - O.2) + N.1 * (O.2 - I.2) in
  (1 / 2) * abs A

theorem area_triangle_NOI_is_5
  (P Q R : ℝ × ℝ)
  (PQ PR QR : ℝ)
  (O I N : ℝ × ℝ)
  (h1 : dist P Q = PQ)
  (h2 : dist Q R = QR)
  (h3 : dist P R = PR)
  (hI : incenter P Q R I)
  (hO : circumcenter P Q R O)
  (hN : is_tangent_circle_to_sides_and_circumcircle P Q R N) :
  triangle_area_NOI P Q R PQ PR QR O I N = 5 := 
sorry

end area_triangle_NOI_is_5_l239_239137


namespace tangent_line_extreme_values_l239_239517

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l239_239517


namespace moles_of_C2H6_formed_l239_239392

-- Define the initial conditions
def initial_moles_H2 : ℕ := 3
def initial_moles_C2H4 : ℕ := 3
def reaction_ratio_C2H4_H2_C2H6 (C2H4 H2 C2H6 : ℕ) : Prop :=
  C2H4 = H2 ∧ C2H4 = C2H6

-- State the theorem to prove
theorem moles_of_C2H6_formed : reaction_ratio_C2H4_H2_C2H6 initial_moles_C2H4 initial_moles_H2 3 :=
by {
  sorry
}

end moles_of_C2H6_formed_l239_239392


namespace part_a_part_b_part_c_part_d_l239_239367

-- (a)
theorem part_a : ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≤ 5 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

-- (b)
theorem part_b : ∃ u v : ℤ, (3 + 2 * Real.sqrt 2)^2 = u + v * Real.sqrt 2 ∧ u^2 - 2 * v^2 = 1 :=
by
  -- proof here
  sorry

-- (c)
theorem part_c : ∀ a b c d : ℤ, a^2 - 2 * b^2 = 1 → (a + b * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) = c + d * Real.sqrt 2
                  → c^2 - 2 * d^2 = 1 :=
by
  -- proof here
  sorry

-- (d)
theorem part_d : ∃ x y : ℤ, y > 100 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

end part_a_part_b_part_c_part_d_l239_239367


namespace line_segments_less_than_sqrt2_l239_239917

theorem line_segments_less_than_sqrt2 (circle_radius : ℝ) (distinct_points : ℕ) (total_pairs : ℕ) :
  circle_radius = 1 ∧ distinct_points = 130 ∧ total_pairs = (130 * 129) / 2 →
  ∃ (l : ℕ), l ≥ 2017 ∧ l ≤ total_pairs ∧
  ∀ (pair_set : finset (finset ℝ)), pair_set.card = total_pairs →
  ∀ (pair ∈ pair_set), (∃ (x y : ℝ) (hx : x ≠ y), 
  x ∈ set.Icc (-1) 1 ∧ y ∈ set.Icc (-1) 1 ∧ pair = {x, y} ∧ 
  (dist x y) < real.sqrt 2) →
  l = {pair ∈ pair_set | (∃ (x y : ℝ) (hx : x ≠ y), 
  x ∈ set.Icc (-1) 1 ∧ y ∈ set.Icc (-1) 1 ∧ pair = {x, y} ∧ 
  (dist x y) < real.sqrt 2)}.card :=
begin
  sorry
end

end line_segments_less_than_sqrt2_l239_239917


namespace prob_of_green_ball_is_25_over_48_l239_239369

def num_red_I := 12
def num_green_I := 6
def num_red_II := 4
def num_green_II := 8
def num_red_III := 3
def num_green_III := 9

def prob_container_I := 1 / 2
def prob_container_II := 1 / 4
def prob_container_III := 1 / 4

def prob_green_ball_I := num_green_I.toRat / (num_red_I + num_green_I).toRat
def prob_green_ball_II := num_green_II.toRat / (num_red_II + num_green_II).toRat
def prob_green_ball_III := num_green_III.toRat / (num_red_III + num_green_III).toRat

def combined_prob_green_ball_I := prob_container_I * prob_green_ball_I
def combined_prob_green_ball_II := prob_container_II * prob_green_ball_II
def combined_prob_green_ball_III := prob_container_III * prob_green_ball_III

def total_prob_green_ball := combined_prob_green_ball_I + combined_prob_green_ball_II + combined_prob_green_ball_III

theorem prob_of_green_ball_is_25_over_48 : total_prob_green_ball = 25 / 48 := 
sorry

end prob_of_green_ball_is_25_over_48_l239_239369


namespace find_A_l239_239980

-- Define the condition as an axiom
axiom A : ℝ
axiom condition : A + 10 = 15 

-- Prove that given the condition, A must be 5
theorem find_A : A = 5 := 
by {
  sorry
}

end find_A_l239_239980


namespace AX_calculation_l239_239342

theorem AX_calculation
  (A B C D X: Type*)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited X]
  (h1: ∀ (p: A), p ∈ circle A B C D)
  (h2: ∀ (p: X), p ∈ diameter A D)
  (h3: distance B X = distance C X)
  (h4: 3 * angle BAC = angle BXC ∧ angle BXC = 36 * π / 180):
  AX = cos (6 * π / 180) * sin (12 * π / 180) * csc (18 * π / 180) :=
by
  sorry

end AX_calculation_l239_239342


namespace focal_length_of_hyperbola_l239_239490

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239490


namespace cubic_cake_icing_l239_239320

theorem cubic_cake_icing (n : ℕ) (h : n = 3) : 
  (let small_cubes := n * n * n in 
   let icing_cubes := (3 * 4 * (n - 1)) / 2 in 
   icing_cubes = 12) :=
by {
  sorry
}

end cubic_cake_icing_l239_239320


namespace ellipse_solution_l239_239430

noncomputable def ellipse_equation := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x / a)^2 + (y / b)^2 = 1)
noncomputable def point_A_on_ellipse (A : ℝ × ℝ) := A = (2, sqrt 2)

-- Mathematical statement that needs to be proven
theorem ellipse_solution :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x / a)^2 + (y / b)^2 = 1) ∧
  ∃ (A : ℝ × ℝ), A = (2, sqrt 2) ∧ ∃ (F1 F2 : ℝ × ℝ),
  (vector.dot_product (A - F2) (F1 - F2) = 0) ∧ (a = sqrt 8) ∧ (b = 2) ∧
  (∃ r : ℝ, r = sqrt (8 / 3) ∧ ∃ (k m : ℝ), 
    (∀ (x1 y1 x2 y2 : ℝ), (y1 = k * x1 + m) ∧ (y2 = k * x2 + m) ∧
    ((x1 / a)^2 + (y1 / b)^2 = 1) ∧ ((x2 / a)^2 + (y2 / b)^2 = 1) ∧
    (x1 * x2 + y1 * y2 = 0) → (x1^2 + y1^2 = r^2))) :=
sorry

end ellipse_solution_l239_239430


namespace number_of_students_l239_239250

theorem number_of_students (n : ℕ) (h1 : 90 - n = n / 2) : n = 60 :=
by
  sorry

end number_of_students_l239_239250


namespace distinct_real_roots_of_quadratic_l239_239879

variable (m : ℝ)

theorem distinct_real_roots_of_quadratic (h1 : 4 + 4 * m > 0) (h2 : m ≠ 0) : m = 1 :=
by
  sorry

end distinct_real_roots_of_quadratic_l239_239879


namespace smallest_solution_l239_239046

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239046


namespace intersection_property_l239_239565

open Set

theorem intersection_property (m : ℝ) : 
  let A := {1, 2, 3}: Set ℝ
  let B := {m, 3, 6}: Set ℝ
  A ∩ B = {2, 3} ↔ m = 2 := 
by
  let A := {1, 2, 3}: Set ℝ
  let B := {m, 3, 6}: Set ℝ
  sorry

end intersection_property_l239_239565


namespace hyperbola_focal_length_is_4_l239_239447

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239447


namespace surveyDSuitableForComprehensiveSurvey_l239_239756

inductive Survey where
| A : Survey
| B : Survey
| C : Survey
| D : Survey

def isComprehensiveSurvey (s : Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => False
  | Survey.C => False
  | Survey.D => True

theorem surveyDSuitableForComprehensiveSurvey : isComprehensiveSurvey Survey.D :=
by
  sorry

end surveyDSuitableForComprehensiveSurvey_l239_239756


namespace quadratic_equation_with_distinct_roots_l239_239587

theorem quadratic_equation_with_distinct_roots 
  (a p q b α : ℝ) 
  (hα1 : α ≠ 0) 
  (h_quad1 : α^2 + a * α + b = 0) 
  (h_quad2 : α^2 + p * α + q = 0) : 
  ∃ x : ℝ, x^2 - (b + q) * (a - p) / (q - b) * x + b * q * (a - p)^2 / (q - b)^2 = 0 :=
by
  sorry

end quadratic_equation_with_distinct_roots_l239_239587


namespace equal_rental_costs_l239_239670

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l239_239670


namespace total_population_l239_239362

theorem total_population :
  let S := 482653 in
  let G := S - 119666 in
  let O := 2 * (S - G) in
  S + G + O = 1084972 := by
    intros
    let S := 482653
    let G := S - 119666
    let O := 2 * (S - G)
    have h1 : S + G + O = 1084972 := sorry
    exact h1

end total_population_l239_239362


namespace tangent_line_extreme_values_l239_239516

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l239_239516


namespace smallest_solution_l239_239055

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239055


namespace smallest_positive_period_of_f_l239_239843

def f (x : ℝ) : ℝ := 1 - 3 * (sin (x + (π / 4)))^2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (T' < T → ¬(∀ x : ℝ, f (x + T') = f x)) := by
  use π
  sorry

end smallest_positive_period_of_f_l239_239843


namespace find_y_l239_239344

theorem find_y :
  (∑ n in Finset.range 1995, n.succ * (1996 - n.succ)) = 1995 * 997 * 333 :=
by
  sorry

end find_y_l239_239344


namespace tangent_periodic_solution_l239_239027

theorem tangent_periodic_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ (Real.tan (n * Real.pi / 180) = Real.tan (345 * Real.pi / 180)) := by
  sorry

end tangent_periodic_solution_l239_239027


namespace cindy_correct_answer_l239_239832

theorem cindy_correct_answer (x : ℝ) : 
  (4 * ((x / 2) - 6) = 24) → (2 * x - 4) / 6 = 22 / 3 :=
by
  intro h
  have hx : x = 24 :=
  begin
    -- Simplify Cindy's incorrect operation
    rw [mul_sub, mul_div_cancel_left] at h,
    { linarith },
    { norm_num }
  end
  -- Use the correct steps according to the instructions
  rw hx
  sorry

end cindy_correct_answer_l239_239832


namespace eval_result_l239_239381

noncomputable def eval_expr : ℚ :=
  64^(-1/3) + 243^(-2/5)

theorem eval_result : eval_expr = 13 / 36 :=
by
  sorry

end eval_result_l239_239381


namespace positive_difference_correct_l239_239677

-- Define the necessary constants
def loan_amount : ℝ := 15000
def rate1 : ℝ := 0.08
def compounding_periods : ℕ := 2
def years1_first_period : ℕ := 7
def years1_second_period : ℕ := 8
def rate2 : ℝ := 0.10
def total_years : ℕ := 15

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Define the simple interest formula
def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

-- Define the total payment for Scheme 1
def total_payment_scheme1 : ℝ :=
  let A1 := compound_interest loan_amount rate1 compounding_periods years1_first_period in
  let half_payment := A1 / 2 in
  let remaining_balance := half_payment in
  let A2 := compound_interest remaining_balance rate1 compounding_periods years1_second_period in
  half_payment + A2

-- Define the total payment for Scheme 2
def total_payment_scheme2 : ℝ :=
  simple_interest loan_amount rate2 total_years

-- Define the positive difference between the total payments
def positive_difference : ℝ :=
  abs (total_payment_scheme2 - total_payment_scheme1)

-- The theorem to prove that the positive difference is $5,447.71
theorem positive_difference_correct : positive_difference ≈ 5447.71 := sorry

end positive_difference_correct_l239_239677


namespace real_solutions_to_gx_eq_g_negx_l239_239241

noncomputable def g (x : ℝ) : ℝ := 
  if x ≠ 0 then some_function x else 0

example : g(1) = g(-1) := by
  sorry

example : g(-1) = g(1) := by
  sorry

theorem real_solutions_to_gx_eq_g_negx (x : ℝ) (hx : x ≠ 0) :
  (g(x) = g(-x)) ↔ (x = 1 ∨ x = -1) := by
  sorry

end real_solutions_to_gx_eq_g_negx_l239_239241


namespace burger_cost_l239_239068

theorem burger_cost (days_in_june : ℕ) (burgers_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_june = 30) (h2 : burgers_per_day = 2) (h3 : total_spent = 720) : 
  total_spent / (burgers_per_day * days_in_june) = 12 :=
by
  -- We will prove this in Lean, but skipping the proof here
  sorry

end burger_cost_l239_239068


namespace expected_value_of_sum_of_marbles_l239_239118

theorem expected_value_of_sum_of_marbles :
  let marbles := [1, 2, 3, 4, 5, 6]
  let sets_of_three := Nat.choose 6 3  -- The number of ways to choose 3 marbles out of 6
  let all_sums := [
    1 + 2 + 3, 1 + 2 + 4, 1 + 2 + 5, 1 + 2 + 6, 1 + 3 + 4,
    1 + 3 + 5, 1 + 3 + 6, 1 + 4 + 5, 1 + 4 + 6, 1 + 5 + 6,
    2 + 3 + 4, 2 + 3 + 5, 2 + 3 + 6, 2 + 4 + 5, 2 + 4 + 6,
    2 + 5 + 6, 3 + 4 + 5, 3 + 4 + 6, 3 + 5 + 6, 4 + 5 + 6
  ]
  let total_sum := List.foldr (.+.) 0 all_sums
  let expected_value := total_sum / sets_of_three
  in expected_value = 10.5 := by
  sorry

end expected_value_of_sum_of_marbles_l239_239118


namespace not_R_A_inter_B_eq_l239_239110

variable (A B : Set ℝ)
def A := {x : ℝ | x^2 - x - 6 ≤ 0}
def B := {x : ℝ | x > 2}

theorem not_R_A_inter_B_eq :
  (λ x, ¬ (x ∈ A ∧ x ∈ B)) = (λ x : ℝ, x ≤ 2 ∨ x > 3) :=
by
  sorry

end not_R_A_inter_B_eq_l239_239110


namespace symmedians_concurrent_l239_239901

noncomputable theory

open_locale classical

variables {A B C D E F K : Type}
variables [linear_ordered_field A] [add_comm_group B] [module A B] [linear_ordered_field C] [add_comm_group D] [module C D]

-- Assuming the existence of triangle ABC
variables (triangle : Type) [is_triangle triangle]

-- Declare the points A, B, C of the triangle
variables {A B C K D E F : Type}

-- Define K as the midpoint of BC
def is_midpoint (K B C : Type) : Prop := sorry

-- Define the line AK as a median of the triangle ABC
def is_median (A K B C : Type) : Prop := sorry

-- Define D as the intersection point of the symmetric line to AK with respect to the bisector of angle A
def is_intersection_point (AK bisector_A BC D: Type) : Prop := sorry

-- Define AD as a symmedian
def is_symmedian (A D B C : Type) : Prop := sorry

-- Define E and F as the intersection points of the symmedians from B and C with AC and AB, respectively
def is_intersection_point_B (symmedian_B AC E: Type) : Prop := sorry
def is_intersection_point_C (symmedian_C AB F: Type) : Prop := sorry

-- Main theorem statement: symmedians are concurrent
theorem symmedians_concurrent [is_triangle triangle] 
  (h1 : is_midpoint K B C) 
  (h2 : is_median A K B C) 
  (h3 : is_intersection_point AK bisector_A BC D) 
  (h4 : is_symmedian A D B C) 
  (h5 : is_intersection_point_B symmedian_B AC E) 
  (h6 : is_intersection_point_C symmedian_C AB F) : 
  are_concurrent A D E F := 
sorry

end symmedians_concurrent_l239_239901


namespace range_of_f_l239_239028

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem range_of_f : set.range (λ x, f x) = set.Icc 2 11 :=
by {
  sorry
}

end range_of_f_l239_239028


namespace sqrt_inequality_l239_239080

theorem sqrt_inequality (n : ℕ) : 
  (n ≥ 0) → (Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n) := 
by
  intro h
  sorry

end sqrt_inequality_l239_239080


namespace find_b_l239_239223

theorem find_b (g : ℝ → ℝ) (g_inv : ℝ → ℝ) (b : ℝ) (h_g_def : ∀ x, g x = 1 / (3 * x + b)) (h_g_inv_def : ∀ x, g_inv x = (1 - 3 * x) / (3 * x)) :
  b = 3 :=
by
  sorry

end find_b_l239_239223


namespace circle_center_sum_l239_239679

noncomputable def circle_center : ℝ × ℝ :=
let a := 6
let b := 8
let c := 15 in
let h := a / 2
let k := b / 2 in
(h, k)

theorem circle_center_sum (a b c : ℝ) :
  a = 6 → b = 8 → c = 15 →
  let h := a / 2 in
  let k := b / 2 in
  h + k = 7 :=
by
  intros ha hb hc
  let h := a / 2
  let k := b / 2
  calc
    h + k = 6 / 2 + 8 / 2 : by { rw [ha, hb] }
    ... = 3 + 4 : by norm_num
    ... = 7 : by norm_num


end circle_center_sum_l239_239679


namespace conjugate_product_equals_2_l239_239503

def z : ℂ := 2 * Complex.I / (1 - Complex.I)
def z_conj : ℂ := conj z

theorem conjugate_product_equals_2 : z_conj * z = 2 := by
  sorry

end conjugate_product_equals_2_l239_239503


namespace total_oranges_picked_l239_239203

theorem total_oranges_picked
  (oranges_first_tree : ℕ := 80)
  (oranges_second_tree : ℕ := 60)
  (oranges_third_tree : ℕ := 120)
  (oranges_fourth_tree : ℕ := 45)
  (oranges_fifth_tree : ℕ := 25)
  (oranges_sixth_tree : ℕ := 97)
  (half : ℕ -> ℕ := λ n, n / 2)
  (twenty_percent_not_ripe : ℕ -> ℕ := λ n, n * 80 / 100) :
  oranges_first_tree 
  + half oranges_second_tree 
  + twenty_percent_not_ripe oranges_third_tree 
  + half oranges_fourth_tree 
  + twenty_percent_not_ripe oranges_fifth_tree 
  + oranges_sixth_tree = 345 := 
by
  sorry

end total_oranges_picked_l239_239203


namespace hyperbola_focal_length_l239_239482

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239482


namespace problem_solution_l239_239507

variable {a b c d m : ℝ}

-- Proposition ①
def prop1 (h1 : a * b > 0) (h2 : a > b) : 1 / a < 1 / b := by
  sorry

-- Proposition ②
def prop2 (h : a > |b|) : a^2 > b^2 := by
  sorry

-- Proposition ④
def prop4 (h1 : 0 < a) (h2 : a < b) (h3 : m > 0) : a / b < (a + m) / (b + m) := by
  sorry

-- Overall statement to be proved
theorem problem_solution :
  (∀ (h1 : a * b > 0) (h2 : a > b), prop1 h1 h2) ∧
  (∀ (h : a > |b|), prop2 h) ∧
  (∀ (h1 : 0 < a) (h2 : a < b) (h3 : m > 0), prop4 h1 h2 h3) := by
  sorry

end problem_solution_l239_239507


namespace weight_of_b_l239_239233

theorem weight_of_b (a b c d : ℝ)
  (h1 : a + b + c + d = 160)
  (h2 : a + b = 50)
  (h3 : b + c = 56)
  (h4 : c + d = 64) :
  b = 46 :=
by sorry

end weight_of_b_l239_239233


namespace monotonicity_intervals_m_range_condition_difference_inequality_l239_239513

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x - m * x + m

theorem monotonicity_intervals (m : ℝ) :
  (∀ x > 0, f x m ≤ f x m) ∧ 
  (∀ x y > 0, x < y → f x m < f y m ↔ m ≤ 0) ∧ 
  (∀ x y > 0, x < y → f x m > f y m → m > 0 ∧ x > (1 / m) ∧ y > (1 / m)) ∧ 
  (∀ x y > 0, x < y → f x m < f y m → m > 0 ∧ x < (1 / m) ∧ y < (1 / m)) := sorry

theorem m_range_condition :
  (∀ x > 0, f x 1 ≤ 0) → (∀ x > 0, f x m ≤ 0 ↔ m = 1) := sorry

theorem difference_inequality (a b : ℝ) (m : ℝ) (h : m = 1) (h0 : 0 < a) (h1 : 0 < b) (h2 : a < b) :
  (f b m - f a m) / (b - a) < 1 / (a * (a + 1)) := sorry

end monotonicity_intervals_m_range_condition_difference_inequality_l239_239513


namespace monthly_growth_rate_l239_239826

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end monthly_growth_rate_l239_239826


namespace janet_has_9_oranges_l239_239618

-- Define conditions
def total_oranges : ℕ := 16
def sharons_oranges : ℕ := 7

-- Define the goal is to find Janet's oranges
def janets_oranges : ℕ := total_oranges - sharons_oranges

-- Prove that Janet has 9 oranges
theorem janet_has_9_oranges : janets_oranges = 9 :=
by 
  unfold janets_oranges
  rw [total_oranges, sharons_oranges]
  norm_num

end janet_has_9_oranges_l239_239618


namespace length_of_major_axis_l239_239773

-- Definitions for the problem conditions
def f1 : ℝ × ℝ := (19, 30)
def f2 : ℝ × ℝ := (59, 80)

-- Calculate the reflection of f1 over the y-axis
def f1' : ℝ × ℝ := (-19, 30)

-- Definition of the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Length of the major axis of the ellipse
def major_axis_length : ℝ :=
  distance f1' f2

theorem length_of_major_axis :
  major_axis_length = real.sqrt 8584 := by
  sorry

end length_of_major_axis_l239_239773


namespace ab_difference_l239_239976

theorem ab_difference (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a + b > 0) : a - b = 2 ∨ a - b = 8 :=
sorry

end ab_difference_l239_239976


namespace knows_all_others_l239_239635

variable (n k : ℕ)
variable (h_pos_n : n > 0) (h_pos_k : k > 0)
variable (h_k : k = 2 * n + 1)

theorem knows_all_others 
  (h_symmetric : ∀ (i j : ℕ), i ≠ j → i.knows j → j.knows i)
  (h_exists : ∀ (G : Finset ℕ), G.card = n → ∃ (x : ℕ), x ∉ G ∧ ∀ y ∈ G, x.knows y) :
  ∃ (p : ℕ), ∀ q : ℕ, q ≠ p → p.knows q :=
sorry

end knows_all_others_l239_239635


namespace ratio_K_L_l239_239164

-- Define the conditions
variable (b : ℕ → ℕ → ℝ) -- each element in the array

-- Calculate sums as per the conditions
def rows_sums (i : ℕ) : ℝ := ∑ j in (Finset.range 100), b i j
def columns_sums (j : ℕ) : ℝ := ∑ i in (Finset.range 50), b i j

-- Define average sums
def K : ℝ := ((∑ i in (Finset.range 50), rows_sums b i) / 50)
def L : ℝ := ((∑ j in (Finset.range 100), columns_sums b j) / 100)

-- State the theorem to be proved
theorem ratio_K_L (b : ℕ → ℕ → ℝ) : K b / L b = 2 := by
  sorry -- proof not required, statement only

end ratio_K_L_l239_239164


namespace number_of_subsets_of_set_1_2_l239_239007

theorem number_of_subsets_of_set_1_2 :
  (set.univ : set (set ({1, 2}))) = 4 := sorry

end number_of_subsets_of_set_1_2_l239_239007


namespace interest_rates_l239_239621

theorem interest_rates (Total_interest : ℝ) (Principal1 Principal2 Rate2_offset : ℝ) :
  Total_interest = 4000 * Rate1 + 8200 * (Rate1 + Rate2_offset) →
  Rate1 = 1159 / 12200 ∧ Rate1 + Rate2_offset = 11 / 100 :=
begin
  intros h1,
  -- The proof is omitted.
  sorry
end

end interest_rates_l239_239621


namespace distance_between_P_and_F2_l239_239921
open Real

theorem distance_between_P_and_F2 (x y c : ℝ) (h1 : c = sqrt 3)
    (h2 : x = -sqrt 3) (h3 : y = 1/2) : 
    sqrt ((sqrt 3 - x) ^ 2 + (0 - y) ^ 2) = 7 / 2 :=
by
  sorry

end distance_between_P_and_F2_l239_239921


namespace number_of_mappings_l239_239931

-- Definitions based on conditions
variables {A : Type} {B : Type} [fintype A] [fintype B] [linear_order B]

-- Assume specific instance sizes
constant (A_set : fin 100 → B)
constant (f_A_B : fin 100 → fin 50)

-- Conditions
axiom f_surjective : function.surjective f_A_B
axiom f_nondec : ∀ i j, i ≤ j → (f_A_B i) ≤ (f_A_B j)

-- Main theorem statement
theorem number_of_mappings : fintype.card {f : fin 100 → fin 50 // function.surjective f ∧ ∀ i j, i ≤ j → f i ≤ f j} = nat.choose 99 49 :=
sorry

end number_of_mappings_l239_239931


namespace find_sr_division_l239_239658

theorem find_sr_division (k : ℚ) (c r s : ℚ)
  (h_c : c = 10)
  (h_r : r = -3 / 10)
  (h_s : s = 191 / 10)
  (h_expr : 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s) :
  s / r = -191 / 3 :=
by
  sorry

end find_sr_division_l239_239658


namespace discount_percentage_is_20_l239_239646

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end discount_percentage_is_20_l239_239646


namespace compute_xy_l239_239727

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end compute_xy_l239_239727


namespace problem1_problem2_l239_239524

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239524


namespace minimum_unsuccessful_placements_l239_239771

noncomputable def unsuccessfulPlacementsOnBoard : Prop :=
  let boardSize := 8
  let cells := fin boardSize × fin boardSize
  let values := ℤ
  ∀ (f : cells → values) (placement : (fin 2 × fin 2) → cells),
    (∀ c, f c = 1 ∨ f c = -1) →
    let unsuccessful := sum (placement <$> (fin 2 × fin 2)) ≠ 0
    in (∃ (crossConfiguration : cells → values),
      (∀ (i j : fin boardSize), i ≠ 0 ∧ i ≠ boardSize - 1 ∧ j ≠ 0 ∧ j ≠ boardSize - 1 →
        let placements := list.prod (list.finRange 2) (list.finRange 2)
        in list.count unsuccessful (placements.map (λ ⟨i, j⟩, (crossConfiguration (i, j)))) ≥ 36)

theorem minimum_unsuccessful_placements : unsuccessfulPlacementsOnBoard := sorry

end minimum_unsuccessful_placements_l239_239771


namespace sum_exterior_angles_const_l239_239343

theorem sum_exterior_angles_const (n : ℕ) (h : n ≥ 3) : 
  ∃ s : ℝ, s = 360 :=
by
  sorry

end sum_exterior_angles_const_l239_239343


namespace encyclopedia_pages_count_l239_239650

theorem encyclopedia_pages_count (digits_used : ℕ) (h : digits_used = 6869) : ∃ pages : ℕ, pages = 1994 :=
by 
  sorry

end encyclopedia_pages_count_l239_239650


namespace sin_double_angle_fourth_quadrant_l239_239971

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239971


namespace percentage_of_water_is_87point5_l239_239573

def percentage_of_water_in_juice (total_juice : ℝ) (puree : ℝ) : ℝ :=
  ((total_juice - puree) / total_juice) * 100

theorem percentage_of_water_is_87point5 :
  percentage_of_water_in_juice 20 2.5 = 87.5 :=
by
  sorry

end percentage_of_water_is_87point5_l239_239573


namespace rabbits_after_n_months_l239_239229

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem rabbits_after_n_months (n : ℕ) : 
  ∃ F : ℕ → ℕ, F 0 = 0 ∧ F 1 = 1 ∧ ∀ n ≥ 2, F n = F (n - 1) + F (n - 2) :=
by
  use fibonacci
  simp [fibonacci]
  sorry

end rabbits_after_n_months_l239_239229


namespace perpendicular_AR_MN_l239_239900

/-
Given these conditions:
1. ABCD is a square.
2. Points M and N are on sides BC and CD, respectively.
3. ∠MAN = 45°.

Prove that:
AR ⊥ MN
-/
theorem perpendicular_AR_MN
  (A B C D M N : Point)
  (H_square : is_square ABCD)
  (H_M_on_BC : on_segment M B C)
  (H_N_on_CD : on_segment N C D)
  (angle_MAN_45 : ∠MAN = 45) : 
  is_perpendicular (line_through A R) (line_through M N) :=
sorry

end perpendicular_AR_MN_l239_239900


namespace right_triangle_incircle_legs_l239_239144

noncomputable def triangle_legs (a b c : ℕ) : Prop :=
  (c * c = a * a + b * b) ∧ c = 17

theorem right_triangle_incircle_legs :
  ∃ a b : ℕ, triangle_legs a b 17 ∧ (a = 8 ∧ b = 15) :=
by {
  have h := show ∃ a b : ℕ, (a + b + 17) / 2 - a = 12 ∧ (a + b + 17) / 2 - b = 5, from sorry,
  obtain ⟨a, b, h₁, h₂⟩ := h,
  use [a, b],
  split,
  exact triangle_legs a b 17,
  split,
  exact h₁,
  exact h₂,
  sorry
}

end right_triangle_incircle_legs_l239_239144


namespace simplest_form_sqrt_8000_l239_239029

-- Definitions based on the conditions given
def is_c_and_d (c d : ℕ) : Prop :=
  (c > 0) ∧ (d > 0) ∧ (d = 1) ∧ (c = 20)

theorem simplest_form_sqrt_8000 : ∃ c d : ℕ, is_c_and_d c d ∧ (c + d = 21) :=
by
  use 20
  use 1
  split
  sorry
  sorry

end simplest_form_sqrt_8000_l239_239029


namespace geometric_sequence_and_general_formula_Sn_inequality_lambda_range_l239_239611

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 1) = 4 * (a n - a (n - 1))

def seq_b (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = 2 * b n ∧ b 1 = 4

def Sn (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range n, a i

def Cn (a : ℕ → ℝ) (C : ℕ → ℝ) (λ : ℝ) : Prop :=
  ∀ n : ℕ, C n = 3 ^ n - λ * (-1) ^ n * a n / (n - 1 / 2)

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) :
  seq_a a →  ∃ (b : ℕ → ℝ), seq_b a b ∧ ∀ n : ℕ, a n = (2 * n - 1) * 2^(n - 1) := 
by
  sorry

theorem Sn_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) :
  seq_a a → Sn a S → ∀ n : ℕ, S n < (n - 1) * 2^(n + 1) + 2 :=
by
  sorry

theorem lambda_range (a : ℕ → ℝ) (C : ℕ → ℝ) (λ : ℝ) :
  seq_a a → Cn a C λ → ∀ n : ℕ, C (n + 1) > C n → λ ∈ Ioo -3 / 2 0 ∨ λ ∈ Ioo 0 1 :=
by
  sorry

end geometric_sequence_and_general_formula_Sn_inequality_lambda_range_l239_239611


namespace total_seashells_found_intact_seashells_found_l239_239722

-- Define the constants for seashells found
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43

-- Define total_intercept
def total_intercept : ℕ := 29

-- Statement that the total seashells found by Tom and Fred is 58
theorem total_seashells_found : tom_seashells + fred_seashells = 58 := by
  sorry

-- Statement that the intact seashells are obtained by subtracting cracked ones
theorem intact_seashells_found : tom_seashells + fred_seashells - total_intercept = 29 := by
  sorry

end total_seashells_found_intact_seashells_found_l239_239722


namespace length_QR_l239_239213

variables (P Q R : Type) [ordered_add_comm_monoid P]
variable PR : P
variable PQ : P
variable QR : P

-- Conditions
axiom h1 : PR = 12
axiom h2 : PQ = 3
axiom h3 : PQ + QR = PR

-- Conclusion
theorem length_QR : QR = 9 :=
by {
  rw [h2, h1, h3],
  -- Subtract PQ = 3 from both sides to find QR
  sorry
}

end length_QR_l239_239213


namespace largest_sum_faces_l239_239324

theorem largest_sum_faces (a b c d e f : ℕ)
  (h_ab : a + b ≤ 7) (h_ac : a + c ≤ 7) (h_ad : a + d ≤ 7) (h_ae : a + e ≤ 7) (h_af : a + f ≤ 7)
  (h_bc : b + c ≤ 7) (h_bd : b + d ≤ 7) (h_be : b + e ≤ 7) (h_bf : b + f ≤ 7)
  (h_cd : c + d ≤ 7) (h_ce : c + e ≤ 7) (h_cf : c + f ≤ 7)
  (h_de : d + e ≤ 7) (h_df : d + f ≤ 7)
  (h_ef : e + f ≤ 7) :
  ∃ x y z, 
  ((x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧ 
   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f)) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (x + y ≤ 7) ∧ (y + z ≤ 7) ∧ (x + z ≤ 7) ∧
  (x + y + z = 9) :=
sorry

end largest_sum_faces_l239_239324


namespace part1_solution_part2_solution_l239_239551

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239551


namespace angle_KPC_right_angle_l239_239712

variable {α : Type*} [MetricSpace α]

-- Definitions and conditions:
variable (A B C E F K P : α)
variable (h_eq_triangle : EquilateralTriangle A B C) 
variable (h_points_on_sides : OnSides E F A B C)
variable (h_equal_segments1 : AE = CF)
variable (h_equal_segments2 : CF = BK)
variable (h_on_extension : OnExtension K A B)
variable (h_midpoint_P : Midpoint P E F)

-- Conclusion to prove:
theorem angle_KPC_right_angle : ∠ (K P C) = 90 :=
by 
  sorry

end angle_KPC_right_angle_l239_239712


namespace find_x_l239_239304

theorem find_x (x : ℝ) : (x / 4 * 5 + 10 - 12 = 48) → (x = 40) :=
by
  sorry

end find_x_l239_239304


namespace min_n_coprime_subset_contains_prime_l239_239178

open Finset
open Nat

def S := (range 2005.succ).filter (λ n, n > 0)

theorem min_n_coprime_subset_contains_prime :
  ∃ n, (∀ A ⊆ S, A.card = n → (∃ p ∈ A, Prime p))
  ∧ (∀ m, (∀ A ⊆ S, A.card = m → (∀ p ∈ A, ¬ Prime p)) → m < n) ∧ 
  n = 16 :=
begin
  sorry
end

end min_n_coprime_subset_contains_prime_l239_239178


namespace coplanar_lambda_l239_239570

theorem coplanar_lambda :
  ∃ (λ : ℝ), let a := (2, -1, 3) in
             let b := (-1, 4, -2) in
             let c := (7, 5, λ) in
             let det := a.1 * (b.2 * c.3 - b.3 * c.2) -
                        a.2 * (b.1 * c.3 - b.3 * c.1) +
                        a.3 * (b.1 * c.2 - b.2 * c.1) in
             det = 0 ↔ λ = 65 / 7 :=
begin
  sorry
end

end coplanar_lambda_l239_239570


namespace regular_octagon_opposite_sides_eq_l239_239819

theorem regular_octagon_opposite_sides_eq (a b c d e f g h : ℤ) 
  (h_equal_angles : true) 
  (h_int_sides : true) 
  (h_sides : List.nth [a, b, c, d, e, f, g, h] 0 = Option.some a ∧
             List.nth [a, b, c, d, e, f, g, h] 1 = Option.some b ∧
             List.nth [a, b, c, d, e, f, g, h] 2 = Option.some c ∧
             List.nth [a, b, c, d, e, f, g, h] 3 = Option.some d ∧
             List.nth [a, b, c, d, e, f, g, h] 4 = Option.some e ∧
             List.nth [a, b, c, d, e, f, g, h] 5 = Option.some f ∧
             List.nth [a, b, c, d, e, f, g, h] 6 = Option.some g ∧
             List.nth [a, b, c, d, e, f, g, h] 7 = Option.some h) :
  a = e ∧ b = f ∧ c = g ∧ d = h :=
sorry

end regular_octagon_opposite_sides_eq_l239_239819


namespace min_height_bounces_l239_239312

noncomputable def geometric_sequence (a r: ℝ) (n: ℕ) : ℝ := 
  a * r^n

theorem min_height_bounces (k : ℕ) : 
  ∀ k, 20 * (2 / 3 : ℝ) ^ k < 3 → k ≥ 7 := 
by
  sorry

end min_height_bounces_l239_239312


namespace OJ_perpendicular_PQ_l239_239089

noncomputable def quadrilateral (A B C D : Point) : Prop := sorry

noncomputable def inscribed (A B C D : Point) : Prop := sorry

noncomputable def circumscribed (A B C D : Point) : Prop := sorry

noncomputable def no_diameter (A B C D : Point) : Prop := sorry

noncomputable def intersection_of_external_bisectors (A B C D : Point) (P : Point) : Prop := sorry

noncomputable def incenter (A B C D J : Point) : Prop := sorry

noncomputable def circumcenter (A B C D O : Point) : Prop := sorry

noncomputable def PQ_perpendicular (O J P Q : Point) : Prop := sorry

theorem OJ_perpendicular_PQ (A B C D P Q J O : Point) :
  quadrilateral A B C D →
  inscribed A B C D →
  circumscribed A B C D →
  no_diameter A B C D →
  intersection_of_external_bisectors A B C D P →
  intersection_of_external_bisectors C D A B Q →
  incenter A B C D J →
  circumcenter A B C D O →
  PQ_perpendicular O J P Q :=
sorry

end OJ_perpendicular_PQ_l239_239089


namespace selling_price_of_car_l239_239218

theorem selling_price_of_car (purchase_price repair_costs : ℝ) (profit_percent : ℝ) :
  purchase_price = 34000 → repair_costs = 12000 → profit_percent = 41.30434782608695 →
  let total_cost := purchase_price + repair_costs in
  let profit := profit_percent / 100 * total_cost in
  let selling_price := total_cost + profit in
  selling_price = 65000 :=
by
  intros hp hr pp
  let total_cost := purchase_price + repair_costs
  let profit := profit_percent / 100 * total_cost
  let selling_price := total_cost + profit
  sorry

end selling_price_of_car_l239_239218


namespace response_percentage_is_50_l239_239199

-- Define the initial number of friends
def initial_friends := 100

-- Define the number of friends Mark kept initially
def kept_friends := 40

-- Define the number of friends Mark contacted
def contacted_friends := initial_friends - kept_friends

-- Define the number of friends Mark has after some responded
def remaining_friends := 70

-- Define the number of friends who responded to Mark's contact
def responded_friends := remaining_friends - kept_friends

-- Define the percentage of contacted friends who responded
def response_percentage := (responded_friends / contacted_friends) * 100

theorem response_percentage_is_50 :
  response_percentage = 50 := by
  sorry

end response_percentage_is_50_l239_239199


namespace ratio_of_projection_l239_239000

theorem ratio_of_projection (x y : ℝ)
  (h : ∀ (x y : ℝ), (∃ x y : ℝ, 
  (3/25 * x + 4/25 * y = x) ∧ (4/25 * x + 12/25 * y = y))) : x / y = 2 / 11 :=
sorry

end ratio_of_projection_l239_239000


namespace crups_are_arogs_and_brafs_l239_239998

variables (Arog Braf Crup Dramp : Type)
variable [Setoid Arog] [Setoid Braf] [Setoid Crup] [Setoid Dramp]

-- Conditions
def all_arogs_are_brafs (a : Arog) : Braf := sorry
def all_crups_are_brafs (c : Crup) : Braf := sorry
def all_dramps_are_arogs (d : Dramp) : Arog := sorry
def all_crups_are_dramps (c : Crup) : Dramp := sorry

-- Proof Problem Statement
theorem crups_are_arogs_and_brafs (c : Crup) : (Arog ∧ Braf) := sorry

end crups_are_arogs_and_brafs_l239_239998


namespace equal_rental_costs_l239_239669

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l239_239669


namespace problem_valid_count_l239_239146

-- Definitions based on conditions
def valid_digit (d : ℕ) := d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def digit_ne_zero (d : ℕ) := d ≠ 0
def digit_ne_one (d : ℕ) := d ≠ 1
def valid_A (a : ℕ) := valid_digit a ∧ digit_ne_zero a ∧ digit_ne_one a
def valid_B (b : ℕ) := valid_digit b ∧ b ≥ 5
def valid_Γ (γ : ℕ) := ∃ b, valid_B b ∧ γ = (2 * b) % 10

-- Additional constraints of the problem
def valid_overline (Б A Я Г : ℕ) := Я = 1 ∧ valid_A A ∧ valid_B Б ∧ valid_Γ Г ∧ (100 ≤ 2 * (10 * Б + A)) ∧ (2 * (10 * Б + A) < 200)

-- Proof goal
theorem problem_valid_count : ∃ (count : ℕ), count = 31 ∧ 
  (∀ (A Б Я Г : ℕ), valid_overline Б A Я Г → (0 < 2 * (10 * Б + A) - (100 + 10 * Г + A)) ∧ (2 * (10 * Б + A) - (100 + 10 * Г + A) < 10)) :=
by
  -- Sorry to skip the proof
  sorry

end problem_valid_count_l239_239146


namespace find_a_l239_239113

noncomputable def m : EuclideanSpace ℝ (Fin 2) := ![-2, 1]
noncomputable def n : EuclideanSpace ℝ (Fin 2) := ![1, 1]

theorem find_a (a : ℝ) (h : (m - 2 • n).dot (a • m + n) = 0) : a = 7 / 9 := by
  sorry

end find_a_l239_239113


namespace father_l239_239938

-- Definitions based on conditions in a)
def cost_MP3_player : ℕ := 120
def cost_CD : ℕ := 19
def total_cost : ℕ := cost_MP3_player + cost_CD
def savings : ℕ := 55
def amount_lacking : ℕ := 64

-- Statement of the proof problem
theorem father's_contribution : (savings + (148:ℕ) - amount_lacking = total_cost) := by
  -- Add sorry to skip the proof
  sorry

end father_l239_239938


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239968

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239968


namespace overlapping_rectangles_perimeter_l239_239262

namespace RectangleOverlappingPerimeter

def length := 7
def width := 3

/-- Prove that the perimeter of the shape formed by overlapping two rectangles,
    each measuring 7 cm by 3 cm, is 28 cm. -/
theorem overlapping_rectangles_perimeter : 
  let total_perimeter := 2 * (length + (2 * width))
  total_perimeter = 28 :=
by
  sorry

end RectangleOverlappingPerimeter

end overlapping_rectangles_perimeter_l239_239262


namespace cos_symmetry_center_l239_239441

theorem cos_symmetry_center : 
  ∀ (m : ℝ), m = π / 2 → (cos m) = 0 := 
by
  intro m
  intro h
  rw h
  simp
  exact cos_pi_div_two
-- sorry


end cos_symmetry_center_l239_239441


namespace floor_abs_l239_239857

namespace Proof

theorem floor_abs {x : ℝ} (hx : x = -3.7) : (⌊|x|⌋ + |⌊x⌋|) = 7 :=
by
  have abs_x : |x| = 3.7 :=
    by rw [hx]; exact abs_of_neg (by norm_num)
  have floor_abs_x : ⌊|x|⌋ = 3 :=
    by rw [abs_x]; exact int.floor_coe_mk (by norm_num)
  have floor_x : ⌊x⌋ = -4 :=
    by rw [hx]; exact int.floor_coe_mk (by norm_num)
  have abs_floor_x : |⌊x⌋| = 4 :=
    by rw [floor_x]; exact abs_of_neg (by norm_num)
  rw [floor_abs_x, abs_floor_x]
  norm_num

end floor_abs_l239_239857


namespace part1_solution_set_part2_range_a_l239_239559

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239559


namespace problem1_problem2_l239_239412

variables (α : ℝ)

def f (α : ℝ) : ℝ := (sin (π / 2 + α) * sin (2 * π - α)) / (cos (-π - α) * sin (3 * π / 2 + α))

-- Problem 1 conditions
axiom third_quadrant (hα : α):  (3 * π) / 2 < α ∧ α < 2 * π
axiom cos_value (hα : α): cos (α - 3 * π / 2) = 1 / 5

-- Problem 1
theorem problem1 (hα : α): (3 * π) / 2 < α ∧ α < 2 * π →
  cos (α - 3 * π / 2) = 1 / 5 →
  f α = sqrt(6) / 12 := sorry

-- Problem 2
theorem problem2 (hα : α): f α = -2 →
  2 * sin α * cos α + cos α ^ 2 = 1 := sorry

end problem1_problem2_l239_239412


namespace length_of_AE_l239_239999

noncomputable def quadrilateral (A B C D E : Type) :=
  ∃ (AB CD AC AE EC areaAED areaBEC : ℝ), AB = 10 ∧ CD = 15 ∧ AC = 18 ∧
  areaAED = 2 * areaBEC ∧ AE + EC = AC ∧ AE = 2 * EC ∧  AE = 12

theorem length_of_AE (A B C D E : Type) : quadrilateral A B C D E → AE = 12 :=
by
  intros h
  obtain ⟨AB, CD, AC, AE, EC, areaAED, areaBEC, AB_eq, CD_eq, AC_eq, area_eq, AE_plus_EC, AE_eq⟩ := h
  have AE_eq_12 : AE = 12 := by { 
    sorry 
  }
  exact AE_eq_12

end length_of_AE_l239_239999


namespace phi_pi_sufficient_not_necessary_l239_239376

theorem phi_pi_sufficient_not_necessary (φ : ℝ) (h1 : ∀ x : ℝ, sin (2 * x + φ) = -sin (-2 * x + φ)) :
  φ = π → ∃ k : ℤ, φ = k * π :=
by
  sorry

end phi_pi_sufficient_not_necessary_l239_239376


namespace circumcircle_eq_midpoint_trajectory_l239_239095

-- Define point structure
structure Point where
  x : ℝ
  y : ℝ

-- Conditions
def A : Point := {x := -1, y := 0}
def B : Point := {x := 3, y := 0}

-- Theorem statement for the first part
theorem circumcircle_eq :
  (x y : ℝ) → (x - 1)^2 + y^2 = 4 → x^2 + y^2 - 2*x - 3 = 0 :=
by
  intros
  everysimp -- Detailed steps skipped, but necessary mathematical simplifications held here
  sorry

-- Theorem statement for the second part
theorem midpoint_trajectory (x y : ℝ) (M : Point) :
  2 * M.x - 3 = x - 2 → 2 * M.y = y → (M.x - 2)^2 + M.y^2 = 1 := 
by
  intros
  everysimp -- Detailed steps skipped, but necessary mathematical simplifications held here
  sorry

end circumcircle_eq_midpoint_trajectory_l239_239095


namespace polynomial_sum_of_squares_l239_239629

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q R : Polynomial ℝ), P = Q^2 + R^2 :=
sorry

end polynomial_sum_of_squares_l239_239629


namespace trapezoid_diagonal_intersection_l239_239772

theorem trapezoid_diagonal_intersection (PQ RS PR : ℝ) (h1 : PQ = 3 * RS) (h2 : PR = 15) :
  ∃ RT : ℝ, RT = 15 / 4 :=
by
  have RT := 15 / 4
  use RT
  sorry

end trapezoid_diagonal_intersection_l239_239772


namespace union_of_M_and_N_l239_239308

open Set

theorem union_of_M_and_N :
  let M := {1, 2, 5}
  let N := {1, 3, 5, 7}
  M ∪ N = {1, 2, 3, 5, 7} :=
by
  sorry

end union_of_M_and_N_l239_239308


namespace no_cube_sum_of_three_consecutive_squares_l239_239388

theorem no_cube_sum_of_three_consecutive_squares :
  ¬∃ x y : ℤ, x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by
  sorry

end no_cube_sum_of_three_consecutive_squares_l239_239388


namespace factorizations_of_4050_l239_239937

theorem factorizations_of_4050 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4050 :=
by
  sorry

end factorizations_of_4050_l239_239937


namespace simplify_fraction_l239_239002

variable {a b c : ℝ}

theorem simplify_fraction (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := 
by
  sorry

end simplify_fraction_l239_239002


namespace calculate_side_a_l239_239409

noncomputable def side_a (b c : ℝ) (A : ℝ) : ℝ :=
  let B := Real.arccos (1 / 7)
  b * Real.sin A / Real.sin B

theorem calculate_side_a :
  side_a 8 3 (Real.pi / 3) ≈ 7.47 :=
by
  sorry

end calculate_side_a_l239_239409


namespace hyperbola_focal_length_l239_239497

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239497


namespace f_prime_neg_one_l239_239134

-- Given conditions and definitions
def f (x : ℝ) (a b c : ℝ) := a * x^4 + b * x^2 + c

def f_prime (x : ℝ) (a b : ℝ) := 4 * a * x^3 + 2 * b * x

-- The theorem we need to prove
theorem f_prime_neg_one (a b c : ℝ) (h : f_prime 1 a b = 2) : f_prime (-1) a b = -2 := by
  sorry

end f_prime_neg_one_l239_239134


namespace range_of_a_l239_239189

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = sqrt (exp x + x - a)) :
  (∃ (x0 y0 : ℝ), y0 = sin x0 ∧ f (f y0) = y0) ↔ 1 ≤ a ∧ a ≤ exp(1) := by
  sorry

end range_of_a_l239_239189


namespace degree_of_poly_l239_239005

def poly : Polynomial ℕ :=
  Polynomial.C 7 * Polynomial.X + 
  2 * Polynomial.X^3 * Polynomial.Y - 
  Polynomial.X^2 * Polynomial.Y - 
  5 * Polynomial.X^3

theorem degree_of_poly : Polynomial.degree poly = 4 :=
by
  sorry

end degree_of_poly_l239_239005


namespace find_common_ratio_l239_239609

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_common_ratio (a1 a4 q : ℝ) (hq : q ^ 3 = 8) (ha1 : a1 = 8) (ha4 : a4 = 64)
  (a_def : is_geometric_sequence (fun n => a1 * q ^ n) q) :
  q = 2 :=
by
  sorry

end find_common_ratio_l239_239609


namespace cos_half_angle_quadrant_l239_239437

theorem cos_half_angle_quadrant 
  (α : ℝ) 
  (h1 : 25 * Real.sin α ^ 2 + Real.sin α - 24 = 0) 
  (h2 : π / 2 < α ∧ α < π) 
  : Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 :=
by
  sorry

end cos_half_angle_quadrant_l239_239437


namespace meet_time_halfway_meet_time_l239_239345

-- Define the times at which Paul and Pierre start their journeys in hours since midnight.
def start_time_paul := 9
def start_time_pierre := 9 + 0.75

-- Define their respective speeds in km/h.
def speed_paul := 15
def speed_pierre := 20

-- Define the distance function for time t in hours.
def distance (speed : ℕ) (time : ℝ) : ℝ := speed * time

-- Define the condition where both Paul and Pierre meet halfway.
theorem meet_time_halfway : ∃ t_meet : ℝ, t_meet = 3 ∧ distance speed_paul t_meet = distance speed_pierre (t_meet - 0.75) := 
by {
  -- Initial known time difference at which Pierre starts, which is 0.75 hours (45 mins).
  let time_diff := 0.75,
  
  -- Define distances to the halfway point.
  let distance_half_paul := distance speed_paul t_meet,
  let distance_half_pierre := distance speed_pierre (t_meet - time_diff),

  -- Using the meet halfway condition, distance_half_paul should be equal to distance_half_pierre.
  have h : distance_half_paul = distance_half_pierre, from sorry,

  -- Solve for t_meet which equates to 3 hours.
  use 3,
  split,
  case left { refl },
  case right { exact h }
}

/-- Final theorem to prove the time at which they meet. -/
theorem meet_time : start_time_paul + 3 = 12 := 
by {
  -- Applying our previous theorem to find the time of meeting.
  obtain ⟨t_meet, t_meet_eq, dist_eq⟩ := meet_time_halfway,
  rw t_meet_eq,
  norm_num
}

end meet_time_halfway_meet_time_l239_239345


namespace cone_height_height_of_cone_l239_239786

theorem cone_height (r_sector : ℝ) (slant_height : ℝ) (arc_length : ℝ) (R : ℝ) 
  (h : ℝ) (sector_count : ℕ) (pi_pos : 0 < Real.pi)
  (r_sector_eq : r_sector = R / sector_count) (circumference_eq : arc_length = 2 * Real.pi * R / sector_count)
  (slant_height_eq : slant_height = R) : 
  h = Real.sqrt (R^2 - r_sector^2) :=
begin
  sorry
end

-- Specific case for given problem
theorem height_of_cone (cone_radius : ℝ) : cone_radius = 2 * Real.sqrt 15 :=
begin
  let r := 8, -- radius of original circle
  let sector_count := 4,
  let arc_length := 2 * Real.pi * 8 / 4, -- arc length of one sector
  let r_base := arc_length / (2 * Real.pi), -- radius of base of the cone
  let slant_height := 8, -- slant height of the cone
  let h := Real.sqrt (slant_height^2 - r_base^2), -- height of the cone
  have r_base_eq : r_base = 2, by norm_num [arc_length, Real.pi],
  have h_eq : h = 2 * Real.sqrt 15, by norm_num [h, slant_height, r_base],
  exact h_eq,
end

end cone_height_height_of_cone_l239_239786


namespace line_intersects_circle_l239_239655

-- Define the problem conditions.
variables {x0 y0 a : ℝ}
hypothesis h_a_pos : a > 0
hypothesis h_outside_circle : x0^2 + y0^2 > a^2

-- Define the statement to be proven.
theorem line_intersects_circle : 
  ∃ d, d = (|a^2| / (real.sqrt (x0^2 + y0^2))) ∧ d < a :=
sorry

end line_intersects_circle_l239_239655


namespace values_of_x_defined_l239_239011

noncomputable def problem_statement (x : ℝ) : Prop :=
  (2 * x - 3 > 0) ∧ (5 - 2 * x > 0)

theorem values_of_x_defined (x : ℝ) :
  problem_statement x ↔ (3 / 2 < x ∧ x < 5 / 2) :=
by sorry

end values_of_x_defined_l239_239011


namespace solve_for_x_l239_239400

theorem solve_for_x (x : ℝ) (h : sqrt (1 - 4 * x) = 5) : x = -6 :=
by
  sorry

end solve_for_x_l239_239400


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239959

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239959


namespace neznaika_discrepancy_l239_239291

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end neznaika_discrepancy_l239_239291


namespace period_tan_3x_l239_239747

theorem period_tan_3x : ∀ x : ℝ, tan (3 * (x + π / 3)) = tan (3 * x) :=
by sorry

end period_tan_3x_l239_239747


namespace big_container_capacity_l239_239313

-- Defining the conditions
variables (C : ℝ)
variables (initially_full : C * 0.40)
variables (added_water : 28)
variables (finally_full : C * 0.75)

-- Stating the problem using the variables
theorem big_container_capacity :
  initially_full + added_water = finally_full → C = 80 :=
begin
  sorry
end

end big_container_capacity_l239_239313


namespace total_area_expanded_dining_area_l239_239803

noncomputable def expanded_dining_area_total : ℝ :=
  let rectangular_area := 35
  let radius := 4
  let semi_circular_area := (1 / 2) * Real.pi * (radius^2)
  rectangular_area + semi_circular_area

theorem total_area_expanded_dining_area :
  expanded_dining_area_total = 60.13272 := by
  sorry

end total_area_expanded_dining_area_l239_239803


namespace required_demand_decrease_l239_239329

theorem required_demand_decrease (P D : ℝ) (hP : P > 0) (hD : D > 0) :
  ((1.20 : ℝ) * P * (D / (1.20 : ℝ)) = P * D) →
  let demand_decrease := (1 - (1 / (1.20 : ℝ))) in
  demand_decrease * 100 = 16.67 :=
by
  intro h₁
  let demand_decrease := (1 - (1 / (1.20 : ℝ)))
  have h₂ : demand_decrease = 0.1667 := by sorry
  have h₃ : 0.1667 * 100 = 16.67 := by sorry
  show demand_decrease * 100 = 16.67 from by sorry

end required_demand_decrease_l239_239329


namespace trigonometric_expression1_trigonometric_expression2_l239_239907

noncomputable def tan_val (α : ℝ) : Prop :=
  Real.tan α = 3

theorem trigonometric_expression1 (α : ℝ) (h : tan_val α) :
  (sqrt 3 * Real.cos (-π - α) - Real.sin (π + α)) /
  (sqrt 3 * Real.cos (π/2 + α) + Real.sin (3*π/2 - α)) = (6 - 5 * sqrt 3) / 13 := sorry

theorem trigonometric_expression2 (α : ℝ) (h : tan_val α) :
  2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 1 = -1 / 10 := sorry

end trigonometric_expression1_trigonometric_expression2_l239_239907


namespace proof_1_proof_2_l239_239908

open Classical

variables (a x x0 : ℝ)
variables (p q : Prop)
variables (U V W: Prop)

/- Definition of proposition p: ∀ x ∈ [1, 2], x^2 - a ≤ 0 -/
def proposition_p := ∀ x ∈ set.Icc 1 2, x^2 - a ≤ 0

/- Definition of proposition q -/
def proposition_q := ∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

/- Proof 1: Prove that if p is true, then a ≥ 4 -/
theorem proof_1 (hp : proposition_p a) : a ≥ 4 :=
sorry

/- Proof 2: Prove that if (p ∨ q) and ¬(p ∧ q) are true, then a ∈ [1, 4) ∪ (-∞, -2] -/
theorem proof_2 (hpq : (proposition_p a ∨ proposition_q a)) (hnpq : ¬(proposition_p a ∧ proposition_q a)) : a ∈ set.Icc 1 4 ∪ set.Iio (-2) :=
sorry

end proof_1_proof_2_l239_239908


namespace Liz_latest_start_time_l239_239196

noncomputable def latest_start_time (turkey_weight : ℕ) (roast_time_per_pound : ℕ) (number_of_turkeys : ℕ) (dinner_time : Time) : Time :=
  Time.sub dinner_time (
    ((turkey_weight * roast_time_per_pound) * number_of_turkeys) / 60
  )

theorem Liz_latest_start_time : 
  latest_start_time 16 15 2 (Time.mk 18 0) = Time.mk 10 0 := 
by
  sorry

end Liz_latest_start_time_l239_239196


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_first_n_terms_l239_239429

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℝ := (1 / 2) ^ n
def c_n (n : ℕ) (a b : ℕ → ℝ) : ℝ := if n % 2 = 1 then a (n / 2 + 1) else b (n / 2)

noncomputable def T_n : ℕ → ℝ
| n => if n % 2 = 0 then
          let k := n / 2
          k * (3 * k + 2) / 8 + 1 - (1 / 2) ^ k
       else
          let k := (n + 1) / 2
          ((n - 1) * (3 * n - 1) / 8 - (1 / 2) ^ (n - 1)) + (3 * k - 1)

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, a_n n = 3 * n - 1 := sorry

theorem geometric_sequence_formula :
  ∀ n : ℕ, b_n n = (1 / 2) ^ n := sorry

theorem sum_of_first_n_terms (n : ℕ) 
  (c := c_n n a_n b_n) :
  T_n n = if n % 2 = 0 then
            let k := n / 2
            (n * (3 * n + 2)) / 8 + 1 - (1 / 2) ^ (n / 2)
         else
            let k := (n + 1) / 2
            ( (n - 1) * (3 * n - 1) / 8 - (1 / 2) ^ (n - 1)) + (3 * k - 1) := sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_first_n_terms_l239_239429


namespace max_lines_with_intersection_angle_l239_239206

theorem max_lines_with_intersection_angle (N : ℕ)
  (h1 : ∀ (i j : ℕ), i ≠ j → 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → ∃ k : ℕ, 1 ≤ k ∧ k ≤ N ∧ 60 = angle_of_lines i j)
  (h2 : ∀ (S : set ℕ), S.card = 15 → ∃ (i j ∈ S), angle_of_lines i j = 60) :
  N ≤ 42 := 
begin
  sorry
end

end max_lines_with_intersection_angle_l239_239206


namespace largest_lcm_18_l239_239270

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l239_239270


namespace opposite_of_two_l239_239703

def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_two : opposite 2 = -2 :=
by
  -- proof skipped
  sorry

end opposite_of_two_l239_239703


namespace find_pxy_l239_239021
open Nat

theorem find_pxy (p x y : ℕ) (hp : Prime p) :
  p^x = y^4 + 4 ↔ (p, x, y) = (5, 1, 1) := sorry

end find_pxy_l239_239021


namespace solve_for_a_l239_239977

theorem solve_for_a (a : ℝ) (h : ∃ x, x = 2 ∧ a * x - 4 * (x - a) = 1) : a = 3 / 2 :=
sorry

end solve_for_a_l239_239977


namespace inequality_solution_set_l239_239665

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 4 - 2 * (2 * x ^ 2 + 7 * x + 3) ^ (1 / 2)) *
            (|x ^ 2 - 4 * x + 2| - |x - 2|) ≤ 0} =
  {x : ℝ | x ∈ Icc (-∞) (-3) ∪ Icc 0 1 ∪ {2} ∪ Icc 3 4} :=
by
  sorry

end inequality_solution_set_l239_239665


namespace vitamin_d_supplements_per_pack_l239_239202

theorem vitamin_d_supplements_per_pack :
  ∃ (x : ℕ), (∀ (n m : ℕ), 7 * n = x * m → 119 <= 7 * n) ∧ (7 * n = 17 * m) :=
by
  -- definition of conditions
  let min_sold := 119
  let vitaminA_per_pack := 7
  -- let x be the number of Vitamin D supplements per pack
  -- the proof is yet to be completed
  sorry

end vitamin_d_supplements_per_pack_l239_239202


namespace avg_weight_of_abc_l239_239232

-- Definitions based on conditions
def weight_a : ℝ := 49
def weight_b : ℝ := 33
def weight_c : ℝ := 53

-- Main proof statement
theorem avg_weight_of_abc : 
  (weight_a + weight_b + weight_c) / 3 = 45 :=
by
  -- Using given conditions to assert the average
  have h1 : weight_a + weight_b = 82 := by sorry
  have h2 : weight_b + weight_c = 86 := by sorry

  -- Using known weights
  rw [h1, h2]
  -- Calculating average
  calc 
    (49 + 33 + 53) / 3 = (135) / 3 := by sorry
  45

end avg_weight_of_abc_l239_239232


namespace sum_binom_mod_500_l239_239174

theorem sum_binom_mod_500 :
  let T := ∑ n in (Finset.range 334), (-1)^n * Nat.choose 1001 (3 * n)
  T % 500 = 6 :=
by
  sorry

end sum_binom_mod_500_l239_239174


namespace train_passing_time_correct_l239_239617

def train_length : ℝ := 350
def train_speed_kmph : ℝ := 85
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
def passing_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_passing_time_correct : 
  passing_time train_length train_speed_mps ≈ 14.82 :=
by 
  sorry

end train_passing_time_correct_l239_239617


namespace no_solution_eq_l239_239071

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l239_239071


namespace find_k_value_l239_239918

theorem find_k_value (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 2) (h4 : Z = K^3)
  (h5 : ∃ n : ℤ, Z = n^6) : K = 16 :=
sorry

end find_k_value_l239_239918


namespace tom_apples_initial_count_l239_239256

theorem tom_apples_initial_count :
  ∃ A : ℕ, (40 - (1 / 4 : ℝ) * 40 + A / 2 = 65) → A = 70 :=
begin
  sorry,
end

end tom_apples_initial_count_l239_239256


namespace find_number_thought_of_l239_239253

theorem find_number_thought_of :
  ∃ x : ℝ, (6 * x^2 - 10) / 3 + 15 = 95 ∧ x = 5 * Real.sqrt 15 / 3 :=
by
  sorry

end find_number_thought_of_l239_239253


namespace cone_height_height_of_cone_l239_239785

theorem cone_height (r_sector : ℝ) (slant_height : ℝ) (arc_length : ℝ) (R : ℝ) 
  (h : ℝ) (sector_count : ℕ) (pi_pos : 0 < Real.pi)
  (r_sector_eq : r_sector = R / sector_count) (circumference_eq : arc_length = 2 * Real.pi * R / sector_count)
  (slant_height_eq : slant_height = R) : 
  h = Real.sqrt (R^2 - r_sector^2) :=
begin
  sorry
end

-- Specific case for given problem
theorem height_of_cone (cone_radius : ℝ) : cone_radius = 2 * Real.sqrt 15 :=
begin
  let r := 8, -- radius of original circle
  let sector_count := 4,
  let arc_length := 2 * Real.pi * 8 / 4, -- arc length of one sector
  let r_base := arc_length / (2 * Real.pi), -- radius of base of the cone
  let slant_height := 8, -- slant height of the cone
  let h := Real.sqrt (slant_height^2 - r_base^2), -- height of the cone
  have r_base_eq : r_base = 2, by norm_num [arc_length, Real.pi],
  have h_eq : h = 2 * Real.sqrt 15, by norm_num [h, slant_height, r_base],
  exact h_eq,
end

end cone_height_height_of_cone_l239_239785


namespace reciprocal_of_repeating_decimal_6_l239_239748

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end reciprocal_of_repeating_decimal_6_l239_239748


namespace min_value_expr_l239_239866

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end min_value_expr_l239_239866


namespace find_x_l239_239598

-- Definitions based on the conditions
def angle1 := 6 * x
def angle2 := 7 * x
def angle3 := 3 * x
def angle4 := 4 * x

-- State the theorem
theorem find_x (h : angle1 + angle2 + angle3 + angle4 = 360) : x = 18 := 
  sorry

end find_x_l239_239598


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239965

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239965


namespace tangent_line_eq_a1_monotonicity_a3_minimum_value_l239_239096

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * |Real.log x - 1|

theorem tangent_line_eq_a1 :
  let a := 1 in
  let f := f in
  let m := Real.deriv_at f 1 1 in
  let t := f 1 1 in
  let tangent_line := m * (x - 1) + t in
  tangent_line = x - y + 1 := 
by 
  admit

theorem monotonicity_a3 :
  let a := 3 in
  let f := f in
  ∀ x : ℝ, 
    (x ∈ (0, Real.sqrt 6 / 2] → f'(x) < 0) ∧ 
    (x ∈ (Real.sqrt 6 / 2, ∞) → f'(x) > 0) := 
by 
  admit

noncomputable def y_min (a : ℝ) : ℝ :=
  if h1 : (0 < a ∧ a ≤ 2) then
    1 + a
  else if h2 : (2 < a ∧ a ≤ 2 * Real.exp 2) then
    3 * a / 2 - a / 2 * Real.log (a / 2)
  else 
    Real.exp (2 : ℝ)

theorem minimum_value :
  ∀ a : ℝ, 0 < a → 
  ∀ x : ℝ, x ∈ [1, ∞) → f x a ≥ y_min a :=
by 
  admit

end tangent_line_eq_a1_monotonicity_a3_minimum_value_l239_239096


namespace line_through_point_with_direction_l239_239861

/-- Define points and direction vectors -/
structure Point where
  x : ℝ
  y : ℝ

structure Vector where
  x : ℝ
  y : ℝ

def slope (v : Vector) : ℝ :=
  v.y / v.x

noncomputable def line_eq (p : Point) (m : ℝ) : ℝ → ℝ :=
  λ x => m * (x - p.x) + p.y

/-- Theorem to prove the equation of the line passing through the point and given direction vector -/
theorem line_through_point_with_direction (p : Point) (v : Vector) :
  ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ (∀ (x y : ℝ), (y - p.y = slope v * (x - p.x)) → (a * x + b * y + c = 0)) :=
by
  use [3, 2, -11]
  constructor
  · left; exact 3 ≠ 0
  · intro x y h
    dsimp [slope, line_eq] at *
    have h1 : y - 1 = (-3 / 2) * (x + 3) := by
      rw [←h); sorry
    dsimp []; sorry
  sorry

end line_through_point_with_direction_l239_239861


namespace setB_is_correct_l239_239433

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l239_239433


namespace sunset_time_correct_l239_239207

def sunrise : Time := ⟨7, 15, .AM⟩
def daylight_length : Duration := ⟨11, 36⟩
def expected_sunset : Time := ⟨6, 51, .PM⟩

theorem sunset_time_correct :
  calculateSunset(sunrise, daylight_length) = expected_sunset := sorry

-- Auxiliary Definitions (Time, Duration, calculateSunset) are usually necessary for Lean
structure Time :=
  (hour : Nat)
  (minute : Nat)
  (period : TimePeriod)

inductive TimePeriod
  | AM | PM

structure Duration :=
  (hours : Nat)
  (minutes : Nat)

noncomputable def calculateSunset (start : Time) (duration : Duration) : Time :=
  sorry

end sunset_time_correct_l239_239207


namespace flowers_twice_duels_l239_239346

variable {K D : ℕ} -- Total number of knights and ladies

-- K_i is the number of knight acquaintances of lady i
variable {K_i : Fin D → ℕ}

-- Define the total duels and total flowers given
def total_duels : ℕ := ∑ i, (K_i i) * (K_i i - 1) / 2
def total_flowers : ℕ := ∑ i, (K_i i) * (K_i i - 1)

theorem flowers_twice_duels (hK : K > 0) (hD : D > 0) :
  total_flowers = 2 * total_duels :=
by
  sorry

end flowers_twice_duels_l239_239346


namespace part1_solution_part2_solution_l239_239549

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239549


namespace find_fraction_l239_239131

theorem find_fraction :
  ∀ (t k : ℝ) (frac : ℝ),
    t = frac * (k - 32) →
    t = 20 → 
    k = 68 → 
    frac = 5 / 9 :=
by
  intro t k frac h_eq h_t h_k
  -- Start from the conditions and end up showing frac = 5/9
  sorry

end find_fraction_l239_239131


namespace min_max_distance_l239_239580

theorem min_max_distance (z : ℂ) (h : complex.abs (z + 2 - 2 * complex.I) = 1) :
  (∃ w : ℂ, w ∈ {z : ℂ | complex.abs (z + 2 - 2 * complex.I) = 1} ∧ complex.abs (w - 2 - 2 * complex.I) = 3) ∧ 
  (∃ w : ℂ, w ∈ {z : ℂ | complex.abs (z + 2 - 2 * complex.I) = 1} ∧ complex.abs (w - 2 - 2 * complex.I) = 5) :=
by
  sorry

end min_max_distance_l239_239580


namespace unit_cube_polygon_perimeter_l239_239415

theorem unit_cube_polygon_perimeter :
  ∀ (points : set (ℝ × ℝ × ℝ)), points.card = 1985 →
  ∃ (subset : finset (ℝ × ℝ × ℝ)), subset.card = 32 ∧
  ∀ (polygon : finset (ℝ × ℝ × ℝ)), polygon ⊆ subset →
  (polygon.perimeter < 8 * real.sqrt 3) :=
by sorry

end unit_cube_polygon_perimeter_l239_239415


namespace find_EF_l239_239152

-- Define the geometric setting and parameters
variables {E F G H P : Type*}
variables (rectangle : ∀ E F G H : Type*, Prop)
variables (P_on_FG : ∀ P : Type*, Prop)
variables (FP PG : ℝ)

def tan_angle_EPH := 2

-- Define the condition of the problem
axiom rectangle_EFGH : rectangle E F G H
axiom P_on_side_FG : P_on_FG P
axiom FP_length : FP = 12
axiom PG_length : PG = 6
axiom tan_EPH_value : tan_angle_EPH = 2

-- State the theorem to prove EF = 12
theorem find_EF (EF : ℝ) 
  (h1 : rectangle_EFGH)
  (h2 : P_on_side_FG)
  (h3 : FP_length)
  (h4 : PG_length)
  (h5 : tan_EPH_value) : EF = 12 :=
sorry

end find_EF_l239_239152


namespace polynomial_abs_sum_l239_239077

theorem polynomial_abs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h : (2*X - 1)^5 = a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0) :
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 243 :=
by
  sorry

end polynomial_abs_sum_l239_239077


namespace hyperbola_focal_length_l239_239481

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239481


namespace hyperbola_focal_length_proof_l239_239456

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239456


namespace contrapositive_of_proposition_l239_239373

theorem contrapositive_of_proposition (a x : ℝ) :
  (∃ x : ℝ, x^2 + (2*a+1)*x + a^2 + 2 ≤ 0) → a ≥ 1 :=
begin
  sorry
end

end contrapositive_of_proposition_l239_239373


namespace number_of_subsets_with_mean_of_remaining_7_l239_239116

open Finset

def original_set : Finset ℕ := (range 14).filter (λ x, 0 < x)

theorem number_of_subsets_with_mean_of_remaining_7 :
  (∑ s in (original_set.ssubsetsLen 3).filter (λ t, (original_set.sum id - t.sum id) = 70), 1) = 5 :=
by
  sorry

end number_of_subsets_with_mean_of_remaining_7_l239_239116


namespace main_inequality_l239_239625

variable {n : ℕ} (x y : Fin n → ℝ)

def is_non_increasing (f : ℕ → ℝ) : Prop :=
  ∀ i j, i < j → f i ≥ f j

def sum_eq_zero (f : Fin n → ℝ) : Prop :=
  ∑ i, f i = 0

def sum_of_squares_eq_one (f : Fin n → ℝ) : Prop :=
  ∑ i, (f i)^2 = 1

theorem main_inequality 
  (hn: n ≥ 2)
  (hx_non_inc: is_non_increasing x)
  (hy_non_inc: is_non_increasing y)
  (hx_sum_zero: sum_eq_zero x)
  (hy_sum_zero: sum_eq_zero y)
  (hx_sum_squares_one: sum_of_squares_eq_one x)
  (hy_sum_squares_one: sum_of_squares_eq_one y):
  ∑ i, (x i * y i - x i * y (n - 1 - i)) ≥ (2 / Real.sqrt (n - 1)) := 
  sorry

end main_inequality_l239_239625


namespace problem_x_plus_y_l239_239993

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem problem_x_plus_y :
  let x := sum_of_integers 30 50
  let y := count_even_integers 30 50
  x + y = 851 :=
by
  let x := sum_of_integers 30 50
  let y := count_even_integers 30 50
  have hx : x = 840 := by sorry
  have hy : y = 11 := by sorry
  exact hx.symm.trans (hy.symm.trans rfl).symm

end problem_x_plus_y_l239_239993


namespace different_product_l239_239812

theorem different_product :
  let P1 := 190 * 80
  let P2 := 19 * 800
  let P3 := 19 * 8 * 10
  let P4 := 19 * 8 * 100
  P3 ≠ P1 ∧ P3 ≠ P2 ∧ P3 ≠ P4 :=
by
  sorry

end different_product_l239_239812


namespace Beth_and_Jan_total_money_l239_239984

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end Beth_and_Jan_total_money_l239_239984


namespace pure_imaginary_a_l239_239890

theorem pure_imaginary_a (a : ℝ) (h : (∃ b : ℝ, complex.abs 0 + b * complex.I = (-a + complex.I) / (1 - complex.I))) : a = -1 := by
  sorry

end pure_imaginary_a_l239_239890


namespace focal_length_of_hyperbola_l239_239472

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239472


namespace segments_divide_longest_side_of_triangle_l239_239091

theorem segments_divide_longest_side_of_triangle :
  ∀ (a b c: ℕ), a = 12 ∧ b = 15 ∧ c = 18 →
  ∃ (x y: ℕ), x + y = c ∧ x = 8 ∧ y = 10 :=
by
  intros a b c h
  cases h with h₁ h_rest
  cases h_rest with h₂ h₃
  use [8, 10]
  split; try split; try assumption
  sorry

end segments_divide_longest_side_of_triangle_l239_239091


namespace rook_tour_possible_iff_even_l239_239684

theorem rook_tour_possible_iff_even (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) : 
  (∃ tour : (ℕ × ℕ) → (ℕ × ℕ), 
    (∀ (k : ℕ), 
      let (i, j) := tour k 
      in (1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n ∧ 
          ((k > 0) → 
           (tour (k - 1) = (i + 1, j) ∨ 
            tour (k - 1) = (i - 1, j) ∨ 
            tour (k - 1) = (i, j + 1) ∨ 
            tour (k - 1) = (i, j - 1)) ∧ 
           (k > 1 → 
            ((tour (k - 2) = (i + 1, j) → tour (k - 1) = (i, j + 1)) ∨ 
             (tour (k - 2) = (i - 1, j) → tour (k - 1) = (i, j - 1)) ∨ 
             (tour (k - 2) = (i, j + 1) → tour (k - 1) = (i - 1, j)) ∨ 
             (tour (k - 2) = (i, j - 1) → tour (k - 1) = (i + 1, j)))) ∧
           (tour (m*n - 1) = tour 0))) ↔ 
  even m ∧ even n :=
sorry

end rook_tour_possible_iff_even_l239_239684


namespace sum_of_solutions_l239_239844

noncomputable def problemCondition (x : ℝ) : Prop :=
  (x^2 - 6*x + 5)^(x^2 - 3*x + 1) = 1

theorem sum_of_solutions : (finset.univ.sum (λ x, if problemCondition x then x else 0)) = 9 :=
sorry

end sum_of_solutions_l239_239844


namespace hyperbola_focal_length_l239_239492

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239492


namespace probability_of_unique_numbers_l239_239332

open_locale big_operators

noncomputable theory

def unique_number_probability : Real :=
  let S := { (A, B, C) ∈ (finset.range 11).product (finset.range 11).product (finset.range 11) | A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ ∃ k m, A = k * B ∧ B = m * C }
  in ((S.card : Real) / (10 * 9 * 8))

theorem probability_of_unique_numbers : unique_number_probability = 1 / 80 := by
  sorry

end probability_of_unique_numbers_l239_239332


namespace probability_point_closer_to_4_is_05_l239_239801

def segment := set.Icc (0 : ℝ) 8
def midpoint (a b : ℝ) := (a + b) / 2

theorem probability_point_closer_to_4_is_05 :
  let m1 := midpoint 0 4
  let m2 := midpoint 4 8
  let closer_segment := set.Icc m1 m2
  let p_closer := (set.volume closer_segment) / (set.volume segment)
  p_closer = 0.5 :=
by
  sorry

end probability_point_closer_to_4_is_05_l239_239801


namespace pyramid_volume_correct_l239_239802

structure RectangularPyramid :=
  (length : ℝ)
  (width : ℝ)
  (edge_length : ℝ)
  (base_area : ℝ := length * width)
  (diagonal_length : ℝ := Real.sqrt (length^2 + width^2))
  (height : ℝ := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2))

noncomputable def pyramid_volume (p : RectangularPyramid) : ℝ :=
  (1 / 3) * p.base_area * p.height

theorem pyramid_volume_correct :
  pyramid_volume {length := 8, width := 10, edge_length := 15} = (160 * Real.sqrt 46) / 3 :=
by
  sorry

end pyramid_volume_correct_l239_239802


namespace enterprise_technical_personnel_avg_and_variance_l239_239816

variables
  (n₁ n₂ : ℕ) -- number of individuals in each group
  (x̄ ȳ : ℝ) -- average ages of each group
  (S₁² S₂² : ℝ) -- variances of each group
  (N : ℕ) -- total number of individuals
  (T : ℝ) -- total combined age
  (S² : ℝ) -- combined variance

-- Given conditions
def conditions : Prop :=
  n₁ = 40 ∧ n₂ = 10 ∧
  x̄ = 35 ∧ ȳ = 45 ∧
  S₁² = 18 ∧ S₂² = 73

-- Statements to prove
def avg_age : ℝ := (n₁ * x̄ + n₂ * ȳ) / N
def variance : ℝ := (1 / N) * ((n₁ - 1) * S₁² + (n₂ - 1) * S₂² + (N / n₁) * (x̄ - (T / N))^2 + (N / n₂) * (ȳ - (T / N))^2)

theorem enterprise_technical_personnel_avg_and_variance
    (h : conditions)
    (hT : T = n₁ * x̄ + n₂ * ȳ)
    (hN : N = n₁ + n₂) :
  avg_age = 37 ∧ variance = 45 :=
by
  sorry

end enterprise_technical_personnel_avg_and_variance_l239_239816


namespace simplified_product_is_one_over_seventy_five_l239_239856

theorem simplified_product_is_one_over_seventy_five :
  (∏ n in (finset.range 148).map (λ x, x + 3), (1 - (1 / n))) = (1 / 75) :=
begin
  sorry,
end

end simplified_product_is_one_over_seventy_five_l239_239856


namespace union_M_N_l239_239566

def M := {x : ℝ | x^2 - 4*x + 3 ≤ 0}
def N := {x : ℝ | Real.log x / Real.log 2 ≤ 1}

theorem union_M_N :
  M ∪ N = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end union_M_N_l239_239566


namespace exists_v_i_l239_239085

-- Define the main statement that needs to be proven
theorem exists_v_i (u : Fin 5 → ℝ) :
  ∃ v : Fin 5 → ℝ, 
     (∀ i : Fin 5, ∃ k : ℤ, u i - v i = k) ∧ 
     (∑ i j in Finset.offDiag (Finset.fin 5), (v i - v j)^2 < 4) :=
sorry

end exists_v_i_l239_239085


namespace part1_f_ge_2_part2_monotonicity_l239_239924

/-- Given the function f(x) = ln(e^x) + 1/x, prove that for all x > 0, f(x) ≥ 2 -/
theorem part1_f_ge_2 (x : ℝ) (hx : x > 0) : ln(exp x) + 1/x ≥ 2 :=
sorry

/-- Let g(x) = e^x/x - a(f(x)) where f(x) = ln(e^x) + 1/x and a > 1. Discuss the monotonicity of g(x) -/
theorem part2_monotonicity (a x : ℝ) (ha1 : 1 < a) :
  let f := λ x : ℝ, ln(exp x) + 1/x in
  let g := λ x : ℝ, exp x / x - a * f x in
  (1 < a ∧ a < exp 1 ∧ x ∈ (0, log a) ∪ (1, ∞) → (g x) > 0) ∧
  (1 < a ∧ a < exp 1 ∧ x ∈ (log a, 1) → (g x) < 0) ∧
  (a = exp 1 ∧ x ∈ (0, ∞) → (g x) > 0) ∧
  (a > exp 1 ∧ x ∈ (0, 1) ∪ (log a, ∞) → (g x) > 0) ∧
  (a > exp 1 ∧ x ∈ (1, log a) → (g x) < 0) :=
sorry

end part1_f_ge_2_part2_monotonicity_l239_239924


namespace one_minus_repeating_six_l239_239359

noncomputable def repeating_six : Real := 2 / 3

theorem one_minus_repeating_six : 1 - repeating_six = 1 / 3 :=
by
  sorry

end one_minus_repeating_six_l239_239359


namespace hyperbola_focal_length_proof_l239_239457

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239457


namespace find_x_reciprocal_of_neg_half_l239_239122

theorem find_x (x : ℝ) (h : -x = -(-3)) : x = -3 :=
by {
  -- placeholder for the proof
  sorry
}

theorem reciprocal_of_neg_half : (-0.5)⁻¹ = -2 :=
by {
  -- placeholder for the proof
  sorry
}

end find_x_reciprocal_of_neg_half_l239_239122


namespace valid_pair_correct_l239_239372

/- Define the problem of finding pairs (m, n) given the conditions -/

theorem valid_pair_correct {m n : ℕ} : 
  ((n = 1 ∧ Nat.gcd m 6 = 1) 
  ∨ (n = 2 ∧ Nat.gcd m 12 = 1)) ↔ (n! ∣ m ∧ ∀ d ∈ {n + 1, n + 2, n + 3}, Nat.gcd n! d ≠ d) :=
by 
  sorry

end valid_pair_correct_l239_239372


namespace all_statements_correct_l239_239997

-- Definitions based on the problem conditions
def population_size : ℕ := 60000
def sample_size : ℕ := 1000
def is_sampling_survey (population_size sample_size : ℕ) : Prop := sample_size < population_size
def is_population (n : ℕ) : Prop := n = 60000
def is_sample (population_size sample_size : ℕ) : Prop := sample_size < population_size
def matches_sample_size (n : ℕ) : Prop := n = 1000

-- Lean problem statement representing the proof that all statements are correct
theorem all_statements_correct :
  is_sampling_survey population_size sample_size ∧
  is_population population_size ∧ 
  is_sample population_size sample_size ∧
  matches_sample_size sample_size := by
  sorry

end all_statements_correct_l239_239997


namespace clever_value_points_count_is_one_l239_239103

def is_clever_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  f x_0 = f' x_0

def count_clever_value_points (fs : List (ℝ → ℝ)) (dfs : List (ℝ → ℝ)) : Nat :=
  List.length (List.filter (λ i, ∃ x_0, is_clever_value_point (fs.nthLe i sorry) (dfs.nthLe i sorry) x_0) (List.range fs.length))

theorem clever_value_points_count_is_one :
  count_clever_value_points
    [λ x => x^2, λ x => Real.exp (-x), λ x => Real.log x, λ x => Real.tan x]
    [λ x => 2 * x, λ x => -Real.exp (-x), λ x => 1 / x, λ x => 1 / (Real.cos x)^2]
  = 1 :=
  sorry

end clever_value_points_count_is_one_l239_239103


namespace smallest_solution_exists_l239_239053

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239053


namespace sin_pi_plus_alpha_l239_239439

theorem sin_pi_plus_alpha (α : ℝ) (h1 : sin (π / 2 + α) = 3/5) (h2 : α ∈ Ioo 0 (π / 2)) :
  sin (π + α) = -4/5 := sorry

end sin_pi_plus_alpha_l239_239439


namespace common_chord_m_eq_17_div_2_l239_239094

theorem common_chord_m_eq_17_div_2 :
  (∃ (k : ℝ), ∀ (x y : ℝ), x^2 + (y - 3/2)^2 = 25/4 ∧ x^2 + y^2 = k ∧
    (m : ℝ), (m = 17/2)) ↔ true := sorry

end common_chord_m_eq_17_div_2_l239_239094


namespace exists_S_l239_239893

/-- Given a plane α and a triangle ABC not parallel to α, 
    and another triangle MNP, there exists a point S such that the lines 
    SA, SB, and SC intersect the plane α at points A', B', and C', forming a triangle 
    A'B'C' congruent to MNP. -/
theorem exists_S (α : AffineSubspace ℝ ℝ^n) (A B C M N P : ℝ^n) 
  (hABC_not_parallel : ¬parallel (affineSpan ℝ {A, B, C}) α) :
  ∃ S : ℝ^n, 
    ∃ A' B' C' : ℝ^n, 
      (A' ≠ B') ∧ (B' ≠ C') ∧ (C' ≠ A') ∧ 
      S ∈ lineThrough ℝ A A' ∧
      S ∈ lineThrough ℝ B B' ∧ 
      S ∈ lineThrough ℝ C C' ∧ 
      A' ∈ α ∧ B' ∈ α ∧ C' ∈ α ∧
      congruent (triangle A' B' C') (triangle M N P) :=
sorry

end exists_S_l239_239893


namespace sin_double_angle_in_fourth_quadrant_l239_239943

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239943


namespace sin_double_angle_fourth_quadrant_l239_239951

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239951


namespace beads_per_necklace_correct_l239_239851
-- Importing the necessary library.

-- Defining the given number of necklaces and total beads.
def number_of_necklaces : ℕ := 11
def total_beads : ℕ := 308

-- Stating the proof goal as a theorem.
theorem beads_per_necklace_correct : (total_beads / number_of_necklaces) = 28 := 
by
  sorry

end beads_per_necklace_correct_l239_239851


namespace sum_of_percentages_l239_239193

theorem sum_of_percentages : 
  let x := 80 + (0.2 * 80)
  let y := 60 - (0.3 * 60)
  let z := 40 + (0.5 * 40)
  x + y + z = 198 := by
  sorry

end sum_of_percentages_l239_239193


namespace sin_double_angle_in_fourth_quadrant_l239_239941

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239941


namespace f_eq_x_l239_239858

noncomputable def f : ℚ+ → ℚ+ := sorry

axiom f_cond1 (x : ℚ+) : f (x + 1) = f x + 1
axiom f_cond2 (x : ℚ+) : f (x^2) = f x ^ 2

theorem f_eq_x (x : ℚ+) : f x = x := by
  sorry

end f_eq_x_l239_239858


namespace triangle_PL_l239_239258

noncomputable def length_PQ : ℝ := 13
noncomputable def length_QR : ℝ := 14
noncomputable def length_RP : ℝ := 15

-- Define PL as PL.
def PL (PL_val : ℝ) : Prop :=
  ∃ PL_val such that 
  PL_val = (25 * Real.sqrt 3) / 2

theorem triangle_PL :
  ∀ (PQ QR RP : ℝ) (PL_val : ℝ),
  PQ = length_PQ →
  QR = length_QR →
  RP = length_RP →
  (PL PL_val) ↔ (PL_val = (25 * Real.sqrt 3) / 2) :=
by
  intros PQ QR RP PL_val hPQ hQR hRP
  rw [hPQ, hQR, hRP]
  sorry

end triangle_PL_l239_239258


namespace find_zeros_l239_239511

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 1/2 then x - 2 / x else x^2 + 2 * x + a - 1

theorem find_zeros (a : ℝ) (h₀ : a > 0):
  ( (a > 2) → (∃ x, f x a = 0 ∧ x = real.sqrt 2) ) ∧
  ( (a = 2) → (∃ x, f x a = 0 ∧ (x = real.sqrt 2 ∨ x = -1)) ) ∧
  ( (0 < a ∧ a < 2) → 
    (∃ x, f x a = 0 ∧ 
      (x = real.sqrt 2 ∨ x = (-1 + real.sqrt (2 - a)) ∨ x = (-1 - real.sqrt (2 - a)))) ) := 
by 
  sorry

end find_zeros_l239_239511


namespace feathers_per_crown_l239_239074

theorem feathers_per_crown (feathers crowns : ℕ) (h_feathers : feathers = 6538) (h_crowns : crowns = 934) :
  Nat.round ((feathers : ℚ) / crowns) = 7 := by
  sorry

end feathers_per_crown_l239_239074


namespace range_of_k_l239_239562

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x ^ 2 + 2 * k * x + 3 ≠ 0) ↔ (0 ≤ k ∧ k < 3) :=
by sorry

end range_of_k_l239_239562


namespace purely_imaginary_iff_real_iff_second_quadrant_iff_l239_239076

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff_l239_239076


namespace polynomial_value_l239_239418

theorem polynomial_value (a : ℝ) (h : a^2 + 2 * a = 1) : 
  2 * a^5 + 7 * a^4 + 5 * a^3 + 2 * a^2 + 5 * a + 1 = 4 :=
by
  sorry

end polynomial_value_l239_239418


namespace min_product_magic_grid_exists_l239_239168

def G : set (ℕ × ℕ) := { p | p.1 ∈ {1, 2, 3, 4, 5, 6} ∧ p.2 ∈ {1, 2, 3, 4, 5, 6} }

def magic_grid (f : ℕ × ℕ → ℤ) : Prop :=
  ∀ (a b : ℕ) (h₁: (a, b) ∈ G) (h₂: (a + 2, b) ∈ G) (h₃: (a + 2, b + 2) ∈ G) 
    (h₄: (a, b + 2) ∈ G),
    f (a, b) + f (a + 2, b + 2) = f (a + 2, b) + f (a, b + 2)

theorem min_product_magic_grid_exists :
  ∃ (f : (ℕ × ℕ) → ℤ), magic_grid f ∧ (∏ (p : ℕ × ℕ) in G, f p) = 6561 :=
sorry

end min_product_magic_grid_exists_l239_239168


namespace johns_gym_time_l239_239163

noncomputable def time_spent_at_gym (day : String) : ℝ :=
  match day with
  | "Monday" => 1 + 0.5
  | "Tuesday" => 40/60 + 20/60 + 15/60
  | "Thursday" => 40/60 + 20/60 + 15/60
  | "Saturday" => 1.5 + 0.75
  | "Sunday" => 10/60 + 50/60 + 10/60
  | _ => 0

noncomputable def total_hours_per_week : ℝ :=
  time_spent_at_gym "Monday" 
  + 2 * time_spent_at_gym "Tuesday" 
  + time_spent_at_gym "Saturday" 
  + time_spent_at_gym "Sunday"

theorem johns_gym_time : total_hours_per_week = 7.4167 := by
  sorry

end johns_gym_time_l239_239163


namespace sequence_term_formula_l239_239641

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 1/2 - 1/2 * a n

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n ≥ 2, a n = r * a (n - 1)

theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 1, S n = 1/2 - 1/2 * a n) →
  (S 1 = 1/2 - 1/2 * a 1) →
  a 1 = 1/3 →
  (∀ n ≥ 2, S n = 1/2 - 1/2 * (a n) → S (n - 1) = 1/2 - 1/2 * (a (n - 1)) → a n = 1/3 * a (n-1)) →
  ∀ n, a n = (1/3)^n :=
by
  intro h1 h2 h3 h4
  sorry

end sequence_term_formula_l239_239641


namespace find_digit_A_l239_239239

theorem find_digit_A (A : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : (2 + A + 3 + A) % 9 = 0) : A = 2 :=
by
  sorry

end find_digit_A_l239_239239


namespace max_marked_vertices_no_rectangle_l239_239745

theorem max_marked_vertices_no_rectangle (n : ℕ) (hn : n = 2016) : 
  ∃ m ≤ n, m = 1009 ∧ 
  ∀ A B C D : Fin n, 
    (A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) ∧ 
    (marked A → marked B → marked C → marked D → 
     ¬is_rectangle A B C D) → 
      (∃ f : Fin n → Bool, marked f ∧ 
      (count_marked f ≤ 1009)) := sorry

end max_marked_vertices_no_rectangle_l239_239745


namespace total_amount_divided_into_two_parts_l239_239781

theorem total_amount_divided_into_two_parts (P1 P2 : ℝ) (annual_income : ℝ) :
  P1 = 1500.0000000000007 →
  annual_income = 135 →
  (P1 * 0.05 + P2 * 0.06 = annual_income) →
  P1 + P2 = 2500.000000000000 :=
by
  intros hP1 hIncome hInterest
  sorry

end total_amount_divided_into_two_parts_l239_239781


namespace barbara_spent_on_other_goods_l239_239356

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l239_239356


namespace relationship_f_g_l239_239636

open Real

noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (a b x : ℝ) : ℝ := a * x + b / x

theorem relationship_f_g (a b x : ℝ) (h1 : a + b = 0) (h2 : f' 1 = g' 1) (hx : x > 1) :
  f x < g a b x := by
  sorry

end relationship_f_g_l239_239636


namespace sin_double_angle_fourth_quadrant_l239_239956

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239956


namespace focal_length_of_hyperbola_l239_239469

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239469


namespace find_a_l239_239424

theorem find_a (a : ℝ) : 
  (a + 3)^2 = (a + 1)^2 + (a + 2)^2 → a = 2 := 
by
  intro h
  -- Proof should go here
  sorry

end find_a_l239_239424


namespace beth_cannot_guarantee_win_l239_239341

theorem beth_cannot_guarantee_win (walls : List ℕ) (h : walls = [5, 3, 3]) : 
  let nim_sum := List.foldr (.⊕.) 0 walls
  nim_sum ≠ 0 :=
by
  simp only [wall, h]
  sorry

end beth_cannot_guarantee_win_l239_239341


namespace student_passes_through_C_l239_239236

-- Define the conditions:
def moves_east_or_south (prob : ℕ → ℕ → Prop) : Prop := 
  ∀ (x y : ℕ), prob (x + 1) y = prob x (y + 1)

def equal_prob_east_south (p : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), p (x + 1) y = 1 / 2 * p x y ∧ p x (y + 1) = 1 / 2 * p x y

def from_A_to_C (path : ℕ) : ℕ :=
  (choose 3 1) -- number of ways to go 2 east and 1 south

def from_C_to_B (path : ℕ) : ℕ :=
  (choose 3 1) -- number of ways to go 1 east and 2 south

def from_A_to_B (path : ℕ) : ℕ :=
  (choose 4 2) -- number of ways to go 3 east and 3 south (note this include total distance 6 steps)

def pass_through_C_prob (P : ℚ) : ℚ :=
  from_A_to_C * from_C_to_B / from_A_to_B

theorem student_passes_through_C :
  pass_through_C_prob = 21 / 32 :=
sorry

end student_passes_through_C_l239_239236


namespace focal_length_of_hyperbola_l239_239473

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239473


namespace rental_cost_equal_mileage_l239_239671

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l239_239671


namespace coordinates_correct_expression_value_correct_l239_239212

noncomputable def B_coordinates (sin_theta : ℝ) (theta_condition : sin_theta = 4/5) : ℝ × ℝ :=
  (-real.sqrt (1 - (sin_theta)^2), sin_theta)

theorem coordinates_correct :
  let B := B_coordinates (4/5) rfl
  in B = (-3/5, 4/5) :=
by
  have sin_theta_val : sin_theta = 4 / 5 := rfl
  let B := B_coordinates (4/5) sin_theta_val
  let ⟨xB, yB⟩ := B
  have y_eq : yB = 4 / 5 := sin_theta_val
  have x_eq : xB = -real.sqrt (1 - (4 / 5 : ℝ)^2) := rfl
  rw [x_eq, y_eq]
  norm_num

theorem expression_value_correct (theta : ℝ) (sin_theta : ℝ) (cos_theta : ℝ) (h1 : sin θ = 4/5)
  (h2 : cos_theta = real.sqrt (1 - (4/5)^2)) :
  (real.sin (real.pi + θ) + 2 * real.sin (real.pi / 2 - θ)) / (2 * real.cos (real.pi - θ))
  = -5 / 3 :=
by
  have sin_pi_theta : real.sin (real.pi + θ) = -real.sin θ := by simp [real.sin_add]
  have sin_half_pi_min_theta : real.sin (real.pi / 2 - θ) = real.cos θ := by simp [real.sin_sub, real.sin_pi_div_two sub θ]
  have cos_pi_min_theta : real.cos (real.pi - θ) = -real.cos θ := by simp [real.cos_sub, real.cos_pi_sub θ]
  rw [sin_pi_theta, sin_half_pi_min_theta, cos_pi_min_theta]
  norm_num
  rw [h1, h2]
  norm_num

example {θ : ℝ} : ∃B : ℝ × ℝ, B = B_coordinates (4/5) rfl ∧
  (real.sin (real.pi + θ) + 2 * real.sin (real.pi / 2 - θ)) / (2 * real.cos (real.pi - θ)) = -5/3 :=
by
  use (-3/5, 4/5)
  split
  exact coordinates_correct
  exact expression_value_correct θ (4/5) (real.sqrt (1 - (4/5)^2)) rfl rfl

end coordinates_correct_expression_value_correct_l239_239212


namespace sequence_G_51_l239_239226

theorem sequence_G_51 :
  ∀ G : ℕ → ℚ, 
  (∀ n : ℕ, G (n + 1) = (3 * G n + 2) / 2) → 
  G 1 = 3 → 
  G 51 = (3^51 + 1) / 2 := by 
  sorry

end sequence_G_51_l239_239226


namespace xy_eq_one_l239_239082

theorem xy_eq_one (x y : ℝ) (h : x + y = (1 / x) + (1 / y) ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_eq_one_l239_239082


namespace triangle_shape_right_angled_l239_239994

noncomputable def is_right_angled_triangle (A B C a b c : ℝ) : Prop :=
  a * Real.cos B + b * Real.cos A = c * Real.sin A →
  (∃ A', A = π / 2 ∧ A' + B + C = π)

theorem triangle_shape_right_angled 
  (A B C a b c : ℝ) 
  (condition : a * Real.cos B + b * Real.cos A = c * Real.sin A) :
  is_right_angled_triangle A B C a b c :=
begin
  sorry
end

end triangle_shape_right_angled_l239_239994


namespace hyperbola_focal_length_is_4_l239_239445

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239445


namespace exists_power_of_two_with_consecutive_zeros_l239_239215

theorem exists_power_of_two_with_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ, 2^n = a * 10^(m + k) + b ∧ 10^(k - 1) ≤ b ∧ b < 10^k ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 :=
sorry

end exists_power_of_two_with_consecutive_zeros_l239_239215


namespace octagon_opposite_sides_equal_l239_239817

theorem octagon_opposite_sides_equal
    (a b c d e f g h : ℕ)
    (equal_angles : ∀ i j, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 → internal_angle i = 135)
    (is_integer_side_lengths : ∀ i, 1 ≤ i ∧ i ≤ 8 → side_length i ∈ ℤ) :
  a = e ∧ b = f ∧ c = g ∧ d = h := 
sorry

end octagon_opposite_sides_equal_l239_239817


namespace find_m_plus_n_plus_d_l239_239139
noncomputable section

-- Define the problem statement
def circle_radius := 36
def chord_length := 66
def intersection_distance := 12
def result_m := 294
def result_n := 81
def result_d := 3

theorem find_m_plus_n_plus_d :
  ∃ (m n d : ℕ), 
    (m * (π : ℝ) - n * real.sqrt d) = 
    (294 * (π : ℝ) - 81 * real.sqrt 3) ∧ 
    m + n + d = 378 :=
by 
  use [result_m, result_n, result_d]
  split
  { rw [real.sqrt_eq_rsqrt, real.sqrt_eq_rsqrt],
    exact rfl, -- establish correct area formula equivalence
    },
  { norm_num, -- establish the sum m + n + d
    }

end find_m_plus_n_plus_d_l239_239139


namespace general_formula_for_a_sum_of_first_2n_terms_of_c_l239_239090

-- Define the sequences and initial conditions
def a : ℕ → ℕ
| 1 := 1
| 2 := 3
| (n+1) := sorry  -- will be defined indirectly through b_n

def b : ℕ → ℕ
| 1 := 2
| n := 2^(n-1)

def c : ℕ → ℕ
| n := if n % 2 = 1 then a n else b n

-- Define the sum of the first n terms of a sequence
def S (f : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum f

-- Define the sum of the first 2n terms of c_n
def T (n : ℕ) : ℕ := S c (2 * n)

-- Main theorem statements based on given conditions and required proofs
theorem general_formula_for_a (n : ℕ) : a n = 2 * n - 1 := sorry

theorem sum_of_first_2n_terms_of_c (n : ℕ) : T n = 2 * n^2 - n - 4/3 + 4^(n+1)/3 := sorry

end general_formula_for_a_sum_of_first_2n_terms_of_c_l239_239090


namespace smallest_solution_exists_l239_239050

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239050


namespace hyperbola_vertex_to_asymptote_distance_l239_239026

theorem hyperbola_vertex_to_asymptote_distance :
  ∀ (x y : ℝ), x^2 - y^2 = 1 → ∃ d : ℝ, d = (|1 * 1 - 1 * 0| / ((1:ℝ)^2 + (1:ℝ)^2).sqrt) ∧ d = (Real.sqrt 2 / 2) :=
by
  rintro x y h
  use (|1 * 1 - 1 * 0| / Real.sqrt ((1:ℝ)^2 + (1:ℝ)^2))
  split
  sorry
  sorry

end hyperbola_vertex_to_asymptote_distance_l239_239026


namespace systematic_sampling_seventh_group_number_l239_239143

theorem systematic_sampling_seventh_group_number 
  (population_size groups : ℕ)
  (x k : ℕ)
  (h_population : population_size = 1000)
  (h_groups : groups = 10)
  (h_x : x = 57)
  (h_k : k = 7) :
  let number_in_kth_group := (x + 33 * k) % 100 in
  600 ≤ 600 + number_in_kth_group ∧ 600 + number_in_kth_group < 700 →
  600 + number_in_kth_group = 688 :=
sorry

lemma systematic_sampling_correct_number 
  : systematic_sampling_seventh_group_number 1000 10 57 7 (by rfl) (by rfl) (by rfl) (by rfl) :=
sorry


end systematic_sampling_seventh_group_number_l239_239143


namespace largest_lcm_18_l239_239269

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l239_239269


namespace min_value_expr_l239_239865

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end min_value_expr_l239_239865


namespace math_problem_l239_239862
noncomputable def lineIntersection := 
  ∃ p : ℝ × ℝ, (2 * p.1 + 3 * p.2 + 5 = 0) ∧ (2 * p.1 + 5 * p.2 + 7 = 0) ∧ p = (-1, -1)

noncomputable def lineEquation := 
  ∀ x y : ℝ, (2 * x + 3 * y + 5 = 0) ∧ (2 * x + 5 * y + 7 = 0) →
  (x = -1 ∧ y = -1) ∧
  ∀ c : ℝ, ((x + 3 * y + c = 0) ∧ (x + 3 * y = 0) ∧ c = 4)

noncomputable def distanceBetweenLines := 
  ∀ a b c1 c2 : ℝ, (a = 1 ∧ b = 3 ∧ c1 = 4 ∧ c2 = 0) →
  (abs (c1 - c2) / (real.sqrt (a ^ 2 + b ^ 2)) = (2 * real.sqrt 10) / 5)

theorem math_problem : lineIntersection → lineEquation → distanceBetweenLines := 
by
  unfold lineIntersection lineEquation distanceBetweenLines
  intros
  split
  split
  sorry

end math_problem_l239_239862


namespace school_committee_count_l239_239406

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def valid_committees_count (total_students total_teachers committee_size : ℕ) : ℕ :=
  let total_people := total_students + total_teachers
  let total_combinations := binom total_people committee_size
  let student_only_combinations := binom total_students committee_size
  total_combinations - student_only_combinations

theorem school_committee_count :
  valid_committees_count 12 3 9 = 4785 :=
by {
  -- Translate the calculation described in the problem to a Lean statement.
  let total_combinations := binom 15 9,
  let student_only_combinations := binom 12 9,
  let valid_com := total_combinations - student_only_combinations,

  -- General binomial coefficient computation simplification is omitted.
  -- Simplify the exact computation here using known binomial identities as required.
  have h1 : binom 15 9 = 5005 := sorry,
  have h2 : binom 12 9 = 220 := sorry,
  
  -- Valid committee count check
  have h3: valid_com = 5005 - 220 := sorry,
  have h4: valid_com = 4785 := by norm_num,
  exact h4,
}

end school_committee_count_l239_239406


namespace length_of_faster_train_l239_239768

theorem length_of_faster_train (speed_fast_train_kmph : ℕ) (speed_slow_train_kmph : ℕ) (time_seconds : ℕ) 
    (h1 : speed_fast_train_kmph = 72) (h2 : speed_slow_train_kmph = 36) (h3 : time_seconds = 37) : 
    length_of_faster_train = 370 :=
by
  --Theorem proof steps go here.
  sorry

end length_of_faster_train_l239_239768


namespace problem1_problem2_l239_239523

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l239_239523


namespace shaded_figure_area_equals_semicircle_area_of_rotation_l239_239390

-- Given definitions from the problem.
def semicircle_area (R : ℝ) : ℝ := (real.pi * R^2) / 2

def shaded_figure_area (R : ℝ) (α : ℝ) (α_deg : α = (45 : ℝ) * real.pi / 180) : ℝ

-- Lean 4 statement of the theorem to prove the area of the shaded figure is equal to semicircle_area R.
theorem shaded_figure_area_equals_semicircle_area_of_rotation (R : ℝ) (α : ℝ) (hα : α = (45 : ℝ) * real.pi / 180) :
  shaded_figure_area R α hα = semicircle_area R :=
sorry

end shaded_figure_area_equals_semicircle_area_of_rotation_l239_239390


namespace part1_part2_part3_l239_239111

open Real InnerProductSpace

variables (a b : E) [InnerProductSpace ℝ E] [Fact (∥a∥ = 4)] [Fact (∥b∥ = 2)] [Fact (inner a b = -4)]

theorem part1 : ∥a + b∥ = 2 * sqrt 3 := sorry

theorem part2 : ∥(3:ℝ) • a - (4:ℝ) • b∥ = 4 * sqrt 19 := sorry

theorem part3 : inner (a - (2:ℝ) • b) (a + b) = 12 := sorry

end part1_part2_part3_l239_239111


namespace money_left_is_40_l239_239200

-- Define the quantities spent by Mildred and Candice, and the total money provided by their mom.
def MildredSpent : ℕ := 25
def CandiceSpent : ℕ := 35
def TotalGiven : ℕ := 100

-- Calculate the total spent and the remaining money.
def TotalSpent := MildredSpent + CandiceSpent
def MoneyLeft := TotalGiven - TotalSpent

-- Prove that the money left is 40.
theorem money_left_is_40 : MoneyLeft = 40 := by
  sorry

end money_left_is_40_l239_239200


namespace find_principal_l239_239396

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h₁ : SI = 8625) (h₂ : R = 50 / 3) (h₃ : T = 3 / 4) :
  SI = (P * R * T) / 100 → P = 69000 := sorry

end find_principal_l239_239396


namespace smallest_period_cos_function_l239_239010

variable (x : ℝ)

def function_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def cos_function (x : ℝ) :=
  Real.cos (4 * x - (5 / 6) * Real.pi)

theorem smallest_period_cos_function :
  function_period cos_function (Real.pi / 2) := by
  sorry

end smallest_period_cos_function_l239_239010


namespace min_students_for_duplicate_borrowings_l239_239807

/-- Given 4 types of books and each student can borrow at most 3 books, 
    the minimum number of students m such that there are at least 
    two students who have borrowed the same type and number of books is 15. -/
theorem min_students_for_duplicate_borrowings
  (books : Finset (Fin 4))
  (max_borrow : ℕ)
  (h_max_borrow : max_borrow = 3) : 
  ∃ m, m = 15 ∧ ∀ (students : Finset (Fin m)) 
  (borrowings : students → Finset books), 
  (∃ i j : students, i ≠ j ∧ borrowings i = borrowings j) :=
by
  sorry

end min_students_for_duplicate_borrowings_l239_239807


namespace eval_powers_of_i_l239_239854

noncomputable def complex_i : ℂ := Complex.I

theorem eval_powers_of_i :
  complex_i^17 + complex_i^203 = 0 :=
by
  -- Definitions based on given conditions
  have h1 : complex_i^1 = complex_i, from Complex.I_one_pow,
  have h2 : complex_i^2 = -1, from Complex.I_sq,
  have h3 : complex_i^3 = -complex_i, from Complex.I_cub_pow,
  have h4 : complex_i^4 = 1, from Complex.I_four_pow,
  -- Evaluation of the expression
  sorry

end eval_powers_of_i_l239_239854


namespace smallest_solution_l239_239058

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239058


namespace trigonometric_identity_l239_239762

theorem trigonometric_identity (α : ℝ) :
  3.404 * (8 * (cos α)^4 - 4 * (cos α)^3 - 8 * (cos α)^2 + 3 * (cos α) + 1) / 
  (8 * (cos α)^4 + 4 * (cos α)^3 - 8 * (cos α)^2 - 3 * (cos α) + 1) = 
  -tan (7 * α / 2) * tan (α / 2) := 
sorry

end trigonometric_identity_l239_239762


namespace remainder_of_x_pow_77_eq_6_l239_239278

theorem remainder_of_x_pow_77_eq_6 (x : ℤ) (h : x^77 % 7 = 6) : x^77 % 7 = 6 :=
by
  sorry

end remainder_of_x_pow_77_eq_6_l239_239278


namespace line_parallel_or_contained_l239_239989

-- Variables and assumptions
variables {Point : Type} {a b : Point → Point → Prop} {α : Point → Prop}

-- Helper definitions for parallelism and containment
def parallel (l1 l2 : Point → Point → Prop) := ∀ P Q R S, l1 P Q → l2 R S → (∃ k, (Q - P) = k * (S - R))
def paralleltop (l : Point → Point → Prop) (p : Point → Prop) := ∀ P Q R, l P Q → p R → (∃ k, (Q - P) = k * (R - S))
def contained (l : Point → Point → Prop) (p : Point → Prop) := ∀ P Q, l P Q → p P ∧ p Q 

-- The main theorem to prove
theorem line_parallel_or_contained (h1 : parallel a b) (h2 : paralleltop b α) :
  paralleltop a α ∨ contained a α :=
sorry

end line_parallel_or_contained_l239_239989


namespace speed_of_stream_l239_239767

theorem speed_of_stream
  (D : ℝ) (v : ℝ)
  (h : D / (72 - v) = 2 * D / (72 + v)) :
  v = 24 := by
  sorry

end speed_of_stream_l239_239767


namespace neznaika_discrepancy_l239_239292

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end neznaika_discrepancy_l239_239292


namespace solve_x_l239_239662

theorem solve_x (
  x : ℂ
) : (x - 4)^6 + (x - 6)^6 = 16 ↔
x = 5 + complex.I ∨
x = 5 - complex.I ∨
x = 5 + complex.I * complex.sqrt (7 + 2 * complex.sqrt 41) ∨
x = 5 - complex.I * complex.sqrt (7 + 2 * complex.sqrt 41) ∨
x = 5 + complex.I * complex.sqrt (7 - 2 * complex.sqrt 41) ∨
x = 5 - complex.I * complex.sqrt (7 - 2 * complex.sqrt 41) := by
sorry

end solve_x_l239_239662


namespace points_form_circle_l239_239877

theorem points_form_circle (s : ℝ) : 
  let x := 2 * s / (1 + s^2),
      y := (1 - s^2) / (1 + s^2) in
  x^2 + y^2 = 1 := 
by
  sorry

end points_form_circle_l239_239877


namespace maximum_vertices_no_rectangle_l239_239734

theorem maximum_vertices_no_rectangle (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, m = 1009 ∧
  ∀ (V : Finset (Fin n)), V.card = m →
  ∀ (v1 v2 v3 v4 : Fin n), {v1, v2, v3, v4}.subset V →
  ¬ (v1.val + v3.val = v2.val + v4.val ∧ v1.val ≠ v2.val ∧ v1.val ≠ v3.val ∧ v1.val ≠ v4.val ∧ v2.val ≠ v3.val ∧ v2.val ≠ v4.val ∧ v3.val ≠ v4.val) :=
sorry

end maximum_vertices_no_rectangle_l239_239734


namespace hyperbola_focal_length_proof_l239_239458

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239458


namespace smallest_solution_l239_239064

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239064


namespace find_number_l239_239988

axiom condition_one (x y : ℕ) : 10 * x + y = 3 * (x + y) + 7
axiom condition_two (x y : ℕ) : x^2 + y^2 - x * y = 10 * x + y

theorem find_number : 
  ∃ (x y : ℕ), (10 * x + y = 37) → (10 * x + y = 3 * (x + y) + 7 ∧ x^2 + y^2 - x * y = 10 * x + y) := 
by 
  sorry

end find_number_l239_239988


namespace central_angle_sector_l239_239105

theorem central_angle_sector (r : ℝ) (h_r_pos : 0 < r) (perimeter_eq : 2 * r + r = 3 * r) : 
  (arc_length : ℝ) (angle : ℝ) (h_arc_length_eq : arc_length = r) (h_angle_eq : angle = arc_length / r) : 
  angle = 1 :=
by 
  sorry

end central_angle_sector_l239_239105


namespace rental_cost_equal_mileage_l239_239672

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l239_239672


namespace find_domain_of_x_l239_239104

variable {f : ℝ → ℝ}
variable {x : ℝ}

theorem find_domain_of_x (h_increasing : ∀ a b : ℝ, a < b → f(a) < f(b))
(h_condition : f 4 < f (2 ^ x)) : x > 2 :=
sorry

end find_domain_of_x_l239_239104


namespace initial_percentage_of_female_workers_l239_239715

noncomputable def initial_percentage_female (E F : ℕ) : ℚ := (F / E : ℚ) * 100

theorem initial_percentage_of_female_workers
  (E : ℕ)  -- The initial number of employees
  (P : ℚ)  -- The initial percentage of female workers
  (hiring_additional_male_workers : E + 28 = 336)  -- Condition 1
  (female_percentage_after_hiring : P = 55)  -- Condition 2
  (total_employees_after_hiring : E + 28 = 336)  -- Condition 3
  (F : ℕ)  -- The initial number of female workers
  (female_workers_after_hiring : F = 0.55 * 336) : 
  initial_percentage_female E F = 59.74 :=
sorry

end initial_percentage_of_female_workers_l239_239715


namespace count_nonzero_terms_l239_239115

noncomputable def polynomial : ℕ :=
  let p1 := (x + 3) * (3 * x^2 + 2 * x + 8)
  let p2 := 4 * (x^4 - 3 * x^3 + x^2)
  let p3 := -2 * (x^3 - 3 * x^2 + 6 * x)
  let result := p1 + p2 + p3
  result.nterms

theorem count_nonzero_terms :
  polynomial = 5 :=
by
  sorry

end count_nonzero_terms_l239_239115


namespace tires_produced_and_sold_l239_239318

-- Define variables and conditions
def cost_per_batch : ℝ := 22500
def cost_per_tire : ℝ := 8
def selling_price_per_tire : ℝ := 20
def profit_per_tire : ℝ := 10.5

-- Define the main theorem that proves the number of tires produced and sold
theorem tires_produced_and_sold : ∃ x : ℝ, profit_per_tire * x = selling_price_per_tire * x - (cost_per_batch + cost_per_tire * x) ∧ x = 15000 :=
by
  use 15000
  split
  sorry

end tires_produced_and_sold_l239_239318


namespace ratio_ad_dc_l239_239591

variable (A B C D : Type) [IsTriangle A B C]
variable (AB AC BC : ℝ)
variable (BD AD DC : ℝ)

-- Define conditions
hypothesis ab_eq_8 : AB = 8
hypothesis bc_eq_10 : BC = 10
hypothesis ac_eq_12 : AC = 12
hypothesis bd_eq_8 : BD = 8
hypothesis d_on_ac : D ∈ LineSegment AC

-- Define the theorem to prove the ratio
theorem ratio_ad_dc (A B C D : Point) [IsTriangle A B C] 
(hab : AB = 8) 
(hbc : BC = 10) 
(hac : AC = 12) 
(hbd : BD = 8) 
(hd_on_ac : D ∈ LineSegment AC) :
  (AD / DC) = (3 / 5) := 
sorry

end ratio_ad_dc_l239_239591


namespace base8_base13_to_base10_sum_l239_239018

-- Definitions for the base 8 and base 13 numbers
def base8_to_base10 (a b c : ℕ) : ℕ := a * 64 + b * 8 + c
def base13_to_base10 (d e f : ℕ) : ℕ := d * 169 + e * 13 + f

-- Constants for the specific numbers in the problem
def num1 := base8_to_base10 5 3 7
def num2 := base13_to_base10 4 12 5

-- The theorem to prove
theorem base8_base13_to_base10_sum : num1 + num2 = 1188 := by
  sorry

end base8_base13_to_base10_sum_l239_239018


namespace smallest_solution_exists_l239_239049

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239049


namespace dragos_wins_l239_239379

variable (S : Set ℕ) [Infinite S]
variable (x : ℕ → ℕ)
variable (M N : ℕ)
variable (p : ℕ)

theorem dragos_wins (h_prime_p : Nat.Prime p) (h_subset_S : p ∈ S) 
  (h_xn_distinct : ∀ i j, i ≠ j → x i ≠ x j) 
  (h_pM_div_xn : ∀ n, n ≥ N → p^M ∣ x n): 
  ∃ N, ∀ n, n ≥ N → p^M ∣ x n :=
sorry

end dragos_wins_l239_239379


namespace domain_of_f_l239_239860

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x^2 + 3*x - 4) / (x^3 - 3*x^2 - 4*x + 12)

def is_defined_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ¬ (x^3 - 3*x^2 - 4*x + 12 = 0)

theorem domain_of_f :
  { x : ℝ | is_defined_at f x } = { x : ℝ | x ∈ set.Ioo (expression by sorry) (3) } ∪ { x : ℝ | x ∈ set.Ioo (3) (expression by sorry) } :=
sorry

end domain_of_f_l239_239860


namespace benny_gave_sandy_books_l239_239357

theorem benny_gave_sandy_books :
  ∀ (Benny_initial Tim_books total_books Benny_after_giving : ℕ), 
    Benny_initial = 24 → 
    Tim_books = 33 →
    total_books = 47 → 
    total_books - Tim_books = Benny_after_giving →
    Benny_initial - Benny_after_giving = 10 :=
by
  intros Benny_initial Tim_books total_books Benny_after_giving
  intros hBenny_initial hTim_books htotal_books hBooks_after
  simp [hBenny_initial, hTim_books, htotal_books, hBooks_after]
  sorry


end benny_gave_sandy_books_l239_239357


namespace nine_op_ten_l239_239935

def op (A B : ℕ) : ℚ := (1 : ℚ) / (A * B) + (1 : ℚ) / ((A + 1) * (B + 2))

theorem nine_op_ten : op 9 10 = 7 / 360 := by
  sorry

end nine_op_ten_l239_239935


namespace focal_length_of_hyperbola_l239_239468

-- Definitions of conditions of the problem
def hyperbola_equation (x y m : ℝ) : Prop := (x^2) / m - y^2 = 1
def asymptote_equation (x y m : ℝ) : Prop := √3 * x + m * y = 0

-- Statement of the mathematically equivalent proof problem in Lean 4
theorem focal_length_of_hyperbola (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∀ x y : ℝ, hyperbola_equation x y m) 
  (h₂ : ∀ x y : ℝ, asymptote_equation x y m) :
  let c := 2 in
  2 * c = 4 := 
by
  sorry

end focal_length_of_hyperbola_l239_239468


namespace friendP_walks_23_km_l239_239725

noncomputable def friendP_distance (v : ℝ) : ℝ :=
  let trail_length := 43
  let speedP := 1.15 * v
  let speedQ := v
  let dQ := trail_length - 23
  let timeP := 23 / speedP
  let timeQ := dQ / speedQ
  if timeP = timeQ then 23 else 0  -- Ensuring that both reach at the same time.

theorem friendP_walks_23_km (v : ℝ) : 
  friendP_distance v = 23 :=
by
  sorry

end friendP_walks_23_km_l239_239725


namespace sailboat_speed_max_power_l239_239688

-- Define constants for the problem.
def B : ℝ := sorry -- Aerodynamic force coefficient (to be provided)
def ρ : ℝ := sorry -- Air density (to be provided)
def S : ℝ := 7 -- sail area in m²
def v0 : ℝ := 6.3 -- wind speed in m/s

-- Define the force formula
def F (v : ℝ) : ℝ := (B * S * ρ * (v0 - v)^2) / 2

-- Define the power formula
def N (v : ℝ) : ℝ := F v * v

-- Define the condition that the power reaches its maximum value at some speed
def N_max : ℝ := sorry -- maximum instantaneous power (to be provided)

-- The proof statement that the speed of the sailboat when the power is maximized is v0 / 3
theorem sailboat_speed_max_power : ∃ v : ℝ, (N v = N_max ∧ v = v0 / 3) := 
  sorry

end sailboat_speed_max_power_l239_239688


namespace latest_time_to_start_roasting_turkeys_l239_239194

theorem latest_time_to_start_roasting_turkeys
  (turkeys : ℕ) 
  (weight_per_turkey : ℕ) 
  (minutes_per_pound : ℕ) 
  (dinner_time_hours : ℕ)
  (dinner_time_minutes : ℕ) 
  (one_at_a_time : turkeys = 2)
  (weight : weight_per_turkey = 16)
  (roasting_time_per_pound : minutes_per_pound = 15)
  (dinner_hours : dinner_time_hours = 18)
  (dinner_minutes : dinner_time_minutes = 0) :
  (latest_start_hours : ℕ) (latest_start_minutes : ℕ) :=
  latest_start_hours = 10 ∧ latest_start_minutes = 0 := 
sorry

end latest_time_to_start_roasting_turkeys_l239_239194


namespace dawn_monthly_payments_l239_239838

theorem dawn_monthly_payments (annual_salary : ℕ) (saved_per_month : ℕ)
  (h₁ : annual_salary = 48000)
  (h₂ : saved_per_month = 400)
  (h₃ : ∀ (monthly_salary : ℕ), saved_per_month = (10 * monthly_salary) / 100):
  annual_salary / saved_per_month = 12 :=
by
  sorry

end dawn_monthly_payments_l239_239838


namespace coeff_x2_term_l239_239024

-- Define the polynomials p(x) and q(x)
def p (x : ℚ) : ℚ[x] := 2 * X^3 + 5 * X^2 - 3 * X + 1
def q (x : ℚ) : ℚ[x] := 3 * X^2 - 9 * X - 5

-- The problem statement
theorem coeff_x2_term :
  polynomial.coeff (p * q) 2 = 5 :=
by
  -- Solution steps should be here, but we add a sorry for now.
  sorry

end coeff_x2_term_l239_239024


namespace number_of_pairs_sum_greater_than_100_l239_239883

theorem number_of_pairs_sum_greater_than_100 :
  (∑ a in finset.range 101, finset.card ((finset.Ico 1 101).filter (λ b, a + b > 100))) = 5050 :=
begin
  -- Variables and setup
  sorry
end

end number_of_pairs_sum_greater_than_100_l239_239883


namespace sin_double_angle_fourth_quadrant_l239_239952

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l239_239952


namespace sailboat_speed_max_power_correct_l239_239687

noncomputable def sailboat_speed_max_power
  (B S ρ v_0 v : ℝ)
  (hS : S = 7)
  (hv0 : v_0 = 6.3)
  (F : ℝ → ℝ := λ v, (B * S * ρ * (v_0 - v) ^ 2) / 2)
  (N : ℝ → ℝ := λ v, F v * v) : Prop :=
  N v = (N (6.3 / 3)) ∧ v = 6.3 / 3

theorem sailboat_speed_max_power_correct :
  sailboat_speed_max_power B S ρ 6.3 2.1 :=
by
  -- Following the above methodology and conditions, we can verify
  -- that the speed v = 6.3 / 3 satisfies the condition when the power
  -- N(v') reaches its maximum value.
  sorry

end sailboat_speed_max_power_correct_l239_239687


namespace parallel_necessary_not_sufficient_l239_239928

-- Define the lines and the parallel condition.
def line1 (a : ℝ) : ℝ × ℝ → ℝ := λ ⟨x, y⟩, a*x + (a+2)*y + 1
def line2 (a : ℝ) : ℝ × ℝ → ℝ := λ ⟨x, y⟩, x + a*y + 2

noncomputable def parallel (f g : ℝ × ℝ → ℝ) := ∀ p1 p2 : ℝ × ℝ, f p1 = 0 → f p2 = 0 → g p1 = 0 → g p2 = 0 → (p1.1 * p2.2 - p1.2 * p2.1) = 0

-- The theorem statement.
theorem parallel_necessary_not_sufficient (a : ℝ) : parallel (line1 a) (line2 a) ↔ (a = 2 ∨ a = -1) := 
by
  sorry

end parallel_necessary_not_sufficient_l239_239928


namespace expected_value_of_three_marbles_l239_239121

-- Define the set of marbles
def marbles := {1, 2, 3, 4, 5, 6}

-- Define the set of possible combinations of drawing 3 marbles
def combinations := marbles.powerset.filter (λ s, s.card = 3)

-- Define the sum of the elements in a set
def sum_set (s : Finset ℕ) : ℕ := s.sum id

-- Define the expected value of the sum of the numbers on the drawn marbles
def expected_value : ℚ :=
  (Finset.sum combinations sum_set : ℚ) / combinations.card

theorem expected_value_of_three_marbles :
  expected_value = 10.05 := sorry

end expected_value_of_three_marbles_l239_239121


namespace part1_solution_set_part2_range_a_l239_239560

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239560


namespace swap_two_numbers_l239_239675

variable (a b : ℤ) (c : ℤ)

theorem swap_two_numbers (h1 : a = 2) (h2 : b = -6) : 
  (let c := a in let a := b in let b := c in (a = -6 ∧ b = 2)) := by 
  sorry

end swap_two_numbers_l239_239675


namespace hyperbola_focal_length_l239_239495

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239495


namespace neither_necessary_nor_sufficient_l239_239125

theorem neither_necessary_nor_sufficient (x : ℝ) : 
  ¬ ((x = 0) ↔ (x^2 - 2 * x = 0) ∧ (x ≠ 0 → x^2 - 2 * x ≠ 0) ∧ (x = 0 → x^2 - 2 * x = 0)) := 
sorry

end neither_necessary_nor_sufficient_l239_239125


namespace find_x_value_l239_239067

theorem find_x_value (x : ℝ) (h : sqrt (x + 7) = 9) : x = 74 := 
by 
  sorry

end find_x_value_l239_239067


namespace total_cost_other_goods_l239_239352

/-- The total cost of the goods other than tuna and water -/
theorem total_cost_other_goods :
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  cost_other = 40 :=
by
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  show cost_other = 40
  sorry

end total_cost_other_goods_l239_239352


namespace parallel_planes_l239_239929

variables {α β γ : Plane} {a b : Line}

-- Definitions for Conditions
def condition1 (α β : Plane) : Prop :=
  ∃ (a : Line), a ⊥ α ∧ a ⊥ β

def condition2 (α β γ : Plane) : Prop :=
  γ ⊥ α ∧ γ ⊥ β

def condition3 (α β : Plane) : Prop :=
  ∃ (a b : Line), a ∥ b ∧ a ⊂ α ∧ b ⊂ β ∧ a ∥ β ∧ b ∥ α

def condition4 (α β : Plane) : Prop :=
  ∃ (a b : Line), skew_lines a b ∧ a ⊂ α ∧ b ⊂ β ∧ a ∥ β ∧ b ∥ α

-- The proof problem statement
theorem parallel_planes (α β : Plane) : 
  condition1 α β ∨ condition4 α β → α ∥ β :=
by sorry

end parallel_planes_l239_239929


namespace sin_alpha_plus_cos_alpha_value_l239_239436

theorem sin_alpha_plus_cos_alpha_value
  (α β : ℝ)
  (h1 : π / 2 < β ∧ β < α ∧ α < 3 * π / 4)
  (h2 : cos (α - β) = 12 / 13)
  (h3 : sin (α + β) = -3 / 5)
  : sin α + cos α = 3 * sqrt 65 / 65 :=
sorry

end sin_alpha_plus_cos_alpha_value_l239_239436


namespace area_PQR_l239_239259

open Real

def P : (ℝ × ℝ) := (4, 0)
def Q : (ℝ × ℝ) := (0, 4)
def R : (ℝ × ℝ) := (8 / 3, 16 / 3)

def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) / 2

theorem area_PQR : area_triangle P Q R = 8 / 3 :=
by
  sorry

end area_PQR_l239_239259


namespace hyperbola_focal_length_l239_239460

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239460


namespace range_of_a_l239_239410

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l239_239410


namespace domain_f_parity_f_odd_range_f_pos_l239_239510

-- Define function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) - log a (1 - x)

-- Domain Theorem
theorem domain_f (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 
  ∀ x, -1 < x ∧ x < 1 ↔ (0 < x + 1) ∧ (0 < 1 - x) :=
sorry

-- Parity Theorem
theorem parity_f_odd (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 
  ∀ x, f a (-x) = -f a x :=
sorry

-- Range Theorem for a > 1
theorem range_f_pos (a : ℝ) (h₀ : a > 1) : 
  ∀ x, 0 < x ∧ x < 1 ↔ 0 < f a x :=
sorry

end domain_f_parity_f_odd_range_f_pos_l239_239510


namespace algebraic_expression_domain_l239_239255

theorem algebraic_expression_domain (x : ℝ) : 
  (x + 2 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 3) := by
  sorry

end algebraic_expression_domain_l239_239255


namespace red_box_position_l239_239283

theorem red_box_position (n : ℕ) (pos_smallest_to_largest : ℕ) (pos_largest_to_smallest : ℕ) 
  (h1 : n = 45) 
  (h2 : pos_smallest_to_largest = 29) 
  (h3 : pos_largest_to_smallest = n - (pos_smallest_to_largest - 1)) :
  pos_largest_to_smallest = 17 := 
  by
    -- This proof is missing; implementation goes here
    sorry

end red_box_position_l239_239283


namespace polynomial_real_zero_l239_239217

noncomputable def P_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), x ^ i / (Nat.factorial i)

theorem polynomial_real_zero (n : ℕ) :
  (even n → ∀ x : ℝ, P_n n x ≠ 0) ∧ (odd n → ∃! x : ℝ, P_n n x = 0) :=
by
  sorry

end polynomial_real_zero_l239_239217


namespace max_marked_vertices_no_rectangle_l239_239730

-- Definitions for the conditions
def regular_polygon (n : ℕ) := n ≥ 3

def no_four_marked_vertices_form_rectangle (n : ℕ) (marked_vertices : Finset ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : ℕ), 
  v1 ∈ marked_vertices ∧ 
  v2 ∈ marked_vertices ∧ 
  v3 ∈ marked_vertices ∧ 
  v4 ∈ marked_vertices → 
  (v1, v2, v3, v4) do not form the vertices of a rectangle in a regular n-gon

-- The theorem we need to prove
theorem max_marked_vertices_no_rectangle (marked_vertices : Finset ℕ) :
  marked_vertices.card ≤ 1009 :=
begin
  assume h1: regular_polygon 2016,
  assume h2: no_four_marked_vertices_form_rectangle 2016 marked_vertices,
  sorry
end

end max_marked_vertices_no_rectangle_l239_239730


namespace clare_remaining_money_l239_239833

theorem clare_remaining_money :
  let money_given := 47
  let bread_cost := 2
  let milk_cost := 2
  let cereal_cost := 3
  let apples_cost := 4
  let bread_qty := 4
  let milk_qty := 2
  let cereal_qty := 3
  let apples_qty := 1
  let total_cost := bread_qty * bread_cost + milk_qty * milk_cost + cereal_qty * cereal_cost + apples_cost * apples_qty
  in money_given - total_cost = 22 :=
by
  let money_given := 47
  let bread_cost := 2
  let milk_cost := 2
  let cereal_cost := 3
  let apples_cost := 4
  let bread_qty := 4
  let milk_qty := 2
  let cereal_qty := 3
  let apples_qty := 1
  let total_cost := bread_qty * bread_cost + milk_qty * milk_cost + cereal_qty * cereal_cost + apples_cost * apples_qty
  have h1 : total_cost = 25 := by sorry
  show money_given - total_cost = 22, from
    by simp [money_given, total_cost, h1]

end clare_remaining_money_l239_239833


namespace sin_double_angle_fourth_quadrant_l239_239969

theorem sin_double_angle_fourth_quadrant (α : ℝ) (h_quadrant : ∃ k : ℤ, -π/2 + 2 * k * π < α ∧ α < 2 * k * π) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_fourth_quadrant_l239_239969


namespace greatest_difference_correct_l239_239979

noncomputable def greatest_difference (x y : ℕ) : ℕ :=
  if 6 < x ∧ x < 10 ∧ Nat.Prime x ∧ 10 < y ∧ y < 17 ∧ ∃ n, y = n^2 then y - x else 0

theorem greatest_difference_correct :
  greatest_difference 7 16 = 9 :=
by
  -- Setting up the conditions
  have x_conditions : 6 < 7 ∧ 7 < 10 ∧ Nat.Prime 7 := by
    exact ⟨by linarith, by linarith, by norm_num⟩

  have y_conditions : 10 < 16 ∧ 16 < 17 ∧ ∃ n, 16 = n^2 := by
    exact ⟨by linarith, by linarith, ⟨4, by norm_num⟩⟩

  -- Let Lean verify the greatest difference under these conditions
  unfold greatest_difference
  rw if_pos,
  { norm_num },
  { exact ⟨x_conditions.left, x_conditions.right.left, x_conditions.right.right, y_conditions.left, y_conditions.right.left, y_conditions.right.right⟩ }

-- Placeholder to make the Lean 4 statement syntactically correct for now
sorry

end greatest_difference_correct_l239_239979


namespace part1_solution_set_part2_range_of_a_l239_239536

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239536


namespace max_perfect_squares_eq_60_l239_239307

open Finset

noncomputable def max_perfect_squares (n : ℕ) (a : Fin n → ℕ) : ℕ :=
if h : n = 100 ∧ (∀ i, a i ∈ range 1 101) ∧ (injective (a : Fin n → ℕ)) then
  let S : Fin n → ℕ := λ k, (range (k.1 + 1)).sum (a ∘ Fin.mk) in
  let perfectSquares := filter (λ x, ∃ m, x = m * m) (image S univ) in
  perfectSquares.card
else 0

theorem max_perfect_squares_eq_60 :
  ∃ (a : Fin 100 → ℕ), max_perfect_squares 100 a = 60 := sorry

end max_perfect_squares_eq_60_l239_239307


namespace sailboat_speed_max_power_l239_239690

-- Define constants for the problem.
def B : ℝ := sorry -- Aerodynamic force coefficient (to be provided)
def ρ : ℝ := sorry -- Air density (to be provided)
def S : ℝ := 7 -- sail area in m²
def v0 : ℝ := 6.3 -- wind speed in m/s

-- Define the force formula
def F (v : ℝ) : ℝ := (B * S * ρ * (v0 - v)^2) / 2

-- Define the power formula
def N (v : ℝ) : ℝ := F v * v

-- Define the condition that the power reaches its maximum value at some speed
def N_max : ℝ := sorry -- maximum instantaneous power (to be provided)

-- The proof statement that the speed of the sailboat when the power is maximized is v0 / 3
theorem sailboat_speed_max_power : ∃ v : ℝ, (N v = N_max ∧ v = v0 / 3) := 
  sorry

end sailboat_speed_max_power_l239_239690


namespace find_ellipse_equation_l239_239815

-- Definitions of the conditions
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a b : ℝ) : ℝ :=
  (Math.sqrt (a^2 - b^2)) / a

def line_equation (x y : ℝ) : Prop :=
  x - y + 1 = 0

def vector_relation (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  3 * (a2 - a1, b2 - b1) = 2 * (-a3, b3 - b2)

-- Main theorem statement
theorem find_ellipse_equation (a b : ℝ) (x y : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) 
  (h_ecc : eccentricity a b = Math.sqrt 3 / 2)
  (h_int : ∃ (pA pB pC : ℝ × ℝ), line_equation pA.1 pA.2 ∧ 
    line_equation pB.1 pB.2 ∧ 
    pC.1 = 0 ∧ 
    3 * (pB.1 - pA.1, pB.2 - pA.2) = 2 * (-pB.1, pC.2 - pB.2)) :
  ellipse_equation x y (2*b) b :=
by {
  sorry
}

end find_ellipse_equation_l239_239815


namespace cindy_correct_answer_l239_239363

theorem cindy_correct_answer (x : ℕ) 
  (h1 : (x - 12) / 2 = 64) : 
  (x - 6) / 4 = 33.5 :=
by
  have hx : x = 140 := by
    linarith
  rw hx
  norm_num
  linarith
  sorry

end cindy_correct_answer_l239_239363


namespace hyperbola_focal_length_is_4_l239_239449

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239449


namespace distance_ratio_gt_9_l239_239606

theorem distance_ratio_gt_9 (points : Fin 1997 → ℝ × ℝ × ℝ) (M m : ℝ) :
  (∀ i j, i ≠ j → dist (points i) (points j) ≤ M) →
  (∀ i j, i ≠ j → dist (points i) (points j) ≥ m) →
  m ≠ 0 →
  M / m > 9 :=
by
  sorry

end distance_ratio_gt_9_l239_239606


namespace max_vertices_no_rectangle_l239_239739

/-- In a regular 2016-sided polygon (2016-gon), prove that the maximum number of vertices 
    that can be marked such that no four marked vertices form the vertices of a rectangle is 1009. -/
theorem max_vertices_no_rectangle (n : ℕ) (h : n = 2016) : 
  ∃ (m : ℕ), m = 1009 ∧ 
    ∀ (marked : finset (fin n)), 
      marked.card ≤ m → 
      (¬ ∃ (a b c d : fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
        (is_rectangle a b c d)) := 
begin 
  sorry 
end

/-- Predicate to determine if four vertices form a rectangle in a regular n-gon. -/
def is_rectangle (a b c d : fin 2016) : Prop := 
  ∃ (k : ℕ), k ∈ finset.range 1008 ∧ 
    ((a = fin.of_nat k) ∧ (b = fin.of_nat (k + 1008)) ∧ 
     (c = fin.of_nat (k + 1008 + 1)) ∧ (d = fin.of_nat (k + 1)) ∨ 
     (a = fin.of_nat (k + 1008)) ∧ (b = fin.of_nat k) ∧ 
     (c = fin.of_nat (k + 1)) ∧ (d = fin.of_nat (k + 1008 + 1)))

end max_vertices_no_rectangle_l239_239739


namespace pack_bangles_l239_239761

theorem pack_bangles
  (dozens_per_box : ℕ)
  (bangles_per_dozen : ℕ)
  (boxes_needed : ℕ) :
  dozens_per_box = 2 →
  bangles_per_dozen = 12 →
  boxes_needed = 20 →
  (boxes_needed * dozens_per_box * bangles_per_dozen) / 2 = 240 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pack_bangles_l239_239761


namespace hyperbola_focal_length_l239_239465

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239465


namespace seq_property_l239_239004

-- Define the sequence {a_n}
def seq (n : ℕ) : ℂ := 
 sorry -- hypothetical definitions based on a1, a2 roots, and recurrence relationships

-- Conditions: a1 and a2 are the roots of the quadratic equation z^2 + iz - 1 = 0
def a1 : ℂ := ( - complex.I + complex.sqrt 3 ) / 2
def a2 : ℂ := ( - complex.I - complex.sqrt 3 ) / 2

-- Recurrence relations for n ≥ 2
def recurrence_relation (n : ℕ) (h : n ≥ 2) : Prop :=
  seq (n + 1) * seq (n - 1) = seq n ^ 2 ∧
  seq (n + 1) + seq (n - 1) = 2 * seq n

-- prove the given property for all natural numbers n:
theorem seq_property (n : ℕ) (h : ∀ k, k ≥ 2 → recurrence_relation k k.ge) :
  seq n^2 + seq (n + 1)^2 + seq (n + 2)^2 = 
  seq n * seq (n + 1) + seq (n + 1) * seq (n + 2) + seq (n + 2) * seq n := 
sorry

end seq_property_l239_239004


namespace find_symmetry_line_eq_l239_239933

theorem find_symmetry_line_eq
    (h1 : ∀ x y : ℝ, x^2 + y^2 = 9)
    (h2 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y - 1 = 0)
    (symm_l : ∀ x y : ℝ,  x^2 + y^2 = 9 ∧ x^2 + y^2 - 4 * x + 4 * y - 1 = 0 → 
                         symmetric_with_respect_to_line x y l) : 
    l = {x : ℝ | x - y - 2 = 0} :=
sorry

end find_symmetry_line_eq_l239_239933


namespace max_marked_vertices_no_rectangle_l239_239731

-- Definitions for the conditions
def regular_polygon (n : ℕ) := n ≥ 3

def no_four_marked_vertices_form_rectangle (n : ℕ) (marked_vertices : Finset ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : ℕ), 
  v1 ∈ marked_vertices ∧ 
  v2 ∈ marked_vertices ∧ 
  v3 ∈ marked_vertices ∧ 
  v4 ∈ marked_vertices → 
  (v1, v2, v3, v4) do not form the vertices of a rectangle in a regular n-gon

-- The theorem we need to prove
theorem max_marked_vertices_no_rectangle (marked_vertices : Finset ℕ) :
  marked_vertices.card ≤ 1009 :=
begin
  assume h1: regular_polygon 2016,
  assume h2: no_four_marked_vertices_form_rectangle 2016 marked_vertices,
  sorry
end

end max_marked_vertices_no_rectangle_l239_239731


namespace maximum_vertices_no_rectangle_l239_239736

theorem maximum_vertices_no_rectangle (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, m = 1009 ∧
  ∀ (V : Finset (Fin n)), V.card = m →
  ∀ (v1 v2 v3 v4 : Fin n), {v1, v2, v3, v4}.subset V →
  ¬ (v1.val + v3.val = v2.val + v4.val ∧ v1.val ≠ v2.val ∧ v1.val ≠ v3.val ∧ v1.val ≠ v4.val ∧ v2.val ≠ v3.val ∧ v2.val ≠ v4.val ∧ v3.val ≠ v4.val) :=
sorry

end maximum_vertices_no_rectangle_l239_239736


namespace part1_solution_set_part2_range_of_a_l239_239539

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239539


namespace smallest_solution_l239_239056

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l239_239056


namespace num_elements_start_with_1_l239_239180

-- Definitions and conditions
def T : set ℕ := {k | ∃ n ∈ (set.Icc 0 1000 : set ℕ), k = 3 ^ n}

-- Axioms for the conditions given in the problem
axiom digits_3_1000 : nat.digits 10 (3 ^ 1000) = 477

-- Theorem statement
theorem num_elements_start_with_1 : (∃ n ∈ T, nat.digits 10 n = 524) :=
sorry

end num_elements_start_with_1_l239_239180


namespace find_point_B_l239_239627

noncomputable def point_A : ℝ × ℝ := (2, 4)

def parabola (x : ℝ) : ℝ := x^2

def tangent_slope (x : ℝ) : ℝ := 2 * x

def normal_slope (x : ℝ) : ℝ := -1 / (tangent_slope x)

def normal_line (x : ℝ) : ℝ × ℝ → ℝ := 
  λ (p : ℝ × ℝ), 
  p.2 + normal_slope p.1 * (x - p.1)

theorem find_point_B :
  let A := (2 : ℝ, 4 : ℝ),
      B := (-9/4 : ℝ, 81/16 : ℝ) in
      normal_line (-9/4) A = parabola (-9/4) → 
      B = (-9/4 : ℝ, 81/16 : ℝ) := 
by
  intros A B h
  sorry

end find_point_B_l239_239627


namespace sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239958

theorem sin_double_angle_neg_of_alpha_in_fourth_quadrant 
  (k : ℤ) (α : ℝ)
  (hα : -π / 2 + 2 * k * π < α ∧ α < 2 * k * π) :
  sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_alpha_in_fourth_quadrant_l239_239958


namespace smallest_munificence_monic_cubic_l239_239873

open Complex

def p (b c : ℂ) : Polynomial ℂ := Polynomial.C (1:ℂ) * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X

def munificence (p : Polynomial ℂ) : ℝ := 
  Real.supSet (Set.range (fun x : ℝ => Complex.abs (p.eval x)))

theorem smallest_munificence_monic_cubic :
  ∀ b c : ℂ → munificence (p b c) ≥ 1 :=
by sorry

end smallest_munificence_monic_cubic_l239_239873


namespace variance_of_binomial_distribution_l239_239987

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_binomial_distribution :
  binomial_variance 10 (2/5) = 12 / 5 :=
by
  sorry

end variance_of_binomial_distribution_l239_239987


namespace part1_solution_set_part2_range_a_l239_239556

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239556


namespace sum_of_sides_equals_triple_base_l239_239154

open Real EuclideanGeometry

-- Define points A, B, and C as variables
variables {A B C P Q : Point}
variables {PQ BC : Line}

-- Define necessary assumptions
-- BP bisects ∠ABC
axiom is_angle_bisector_bp : is_angle_bisector B P C A
-- CP bisects ∠BCA
axiom is_angle_bisector_cp : is_angle_bisector C P A B
-- PQ is perpendicular to BC
axiom is_perpendicular_pq : is_perpendicular PQ BC

-- Condition BQ * QC = 2PQ^2
axiom bq_qc_condition : BQ.distance * QC.distance = 2 * (PQ.length ^ 2)

-- Theorem to prove
theorem sum_of_sides_equals_triple_base :
  AB.length + AC.length = 3 * BC.length :=
by
  sorry

end sum_of_sides_equals_triple_base_l239_239154


namespace two_snakes_swallow_termination_l239_239135

structure SnakesSwallowing :=
(snake1_swallowing_snake2 : Prop)
(snake2_swallowing_snake1 : Prop)
(loop_formed : Prop)
(loop_decreasing : Prop)

theorem two_snakes_swallow_termination (s : SnakesSwallowing) :
  (s.snake1_swallowing_snake2 ∧ s.snake2_swallowing_snake1 ∧ s.loop_formed ∧ s.loop_decreasing) →
  ¬(s.snake1_swallowing_snake2 ∧ s.snake2_swallowing_snake1 ∧ False) :=
by { intro h, sorry }

end two_snakes_swallow_termination_l239_239135


namespace find_prime_triples_l239_239022

theorem find_prime_triples :
  ∃ p q n : ℕ,
    Nat.Prime p ∧
    Nat.Prime q ∧
    p > 0 ∧
    q > 0 ∧
    n > 0 ∧
    (p * (p + 1) + q * (q + 1) = n * (n + 1)) ∧
    ((p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 3 ∧ q = 5 ∧ n = 6)) :=
begin
  sorry
end

end find_prime_triples_l239_239022


namespace part1_solution_set_part2_range_a_l239_239557

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l239_239557


namespace inequality_solution_correct_l239_239423

variable (f : ℝ → ℝ)

def f_one : Prop := f 1 = 1

def f_prime_half : Prop := ∀ x : ℝ, (deriv f x) > (1 / 2)

def inequality_solution_set : Prop := ∀ x : ℝ, f (x^2) < (x^2 / 2 + 1 / 2) ↔ -1 < x ∧ x < 1

theorem inequality_solution_correct (h1 : f_one f) (h2 : f_prime_half f) : inequality_solution_set f := sorry

end inequality_solution_correct_l239_239423


namespace log_expression_equality_l239_239829

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_equality :
  Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + (log_base 2 5) * (log_base 5 8) = 5 := by
  sorry

end log_expression_equality_l239_239829


namespace largest_lcm_l239_239272

theorem largest_lcm :
  let l4 := Nat.lcm 18 3
  let l5 := Nat.lcm 18 6
  let l6 := Nat.lcm 18 9
  let l7 := Nat.lcm 18 12
  let l8 := Nat.lcm 18 15
  let l9 := Nat.lcm 18 18
  in max (max (max (max (max l4 l5) l6) l7) l8) l9 = 90 := by
    sorry

end largest_lcm_l239_239272


namespace part1_solution_set_part2_range_of_a_l239_239538

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239538


namespace smallest_solution_l239_239037

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239037


namespace subset_property_l239_239184

theorem subset_property (n : ℕ) (h1 : n > 6) (X : finset ℕ) (h2 : X.card = n) (A : finset (finset ℕ)) (h3 : ∀ B ∈ A, B.card = 5) (m : ℕ) (h4 : A.card = m) (h5 : m > (n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15)) / 600) :
  ∃ (i1 i2 i3 i4 i5 i6 : ℕ), 1 ≤ i1 ∧ i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧ i6 ≤ m ∧ (finset.bUnion (finset.filter (λ i, i = i1 ∨ i = i2 ∨ i = i3 ∨ i = i4 ∨ i = i5 ∨ i = i6) A)).card = 6 :=
sorry

end subset_property_l239_239184


namespace beth_jan_total_money_l239_239982

theorem beth_jan_total_money (beth_money jan_money : ℕ)
    (h1 : beth_money + 35 = 105)
    (h2 : jan_money - 10 = beth_money) : beth_money + jan_money = 150 :=
begin
  sorry
end

end beth_jan_total_money_l239_239982


namespace base_five_product_l239_239831

theorem base_five_product (a b : ℕ) (h1 : a = 1324) (h2 : b = 32) :
    let sum_ab := 231 in  -- Sum of 1324_5 + 32_5
    let prod_ab := 24122 in  -- Resulting product in base 5
    base_five_product (sum_ab * b) prod_ab = 24122 
:=
sorry

end base_five_product_l239_239831


namespace max_marked_vertices_no_rectangle_l239_239743

theorem max_marked_vertices_no_rectangle (n : ℕ) (hn : n = 2016) : 
  ∃ m ≤ n, m = 1009 ∧ 
  ∀ A B C D : Fin n, 
    (A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) ∧ 
    (marked A → marked B → marked C → marked D → 
     ¬is_rectangle A B C D) → 
      (∃ f : Fin n → Bool, marked f ∧ 
      (count_marked f ≤ 1009)) := sorry

end max_marked_vertices_no_rectangle_l239_239743


namespace addition_terms_correct_l239_239128

def first_seq (n : ℕ) : ℕ := 2 * n + 1
def second_seq (n : ℕ) : ℕ := 5 * n - 1

theorem addition_terms_correct :
  first_seq 10 = 21 ∧ second_seq 10 = 49 ∧
  first_seq 80 = 161 ∧ second_seq 80 = 399 :=
by
  sorry

end addition_terms_correct_l239_239128


namespace mode_is_98_l239_239708

-- Define the stem-and-leaf data as an array of arrays (rows).
def stem_and_leaf_plot : List (Nat × List Nat) := [
  (6, [0, 5, 5]),
  (7, [2, 3, 3, 3, 5, 6]),
  (8, [0, 4, 6, 6, 6, 7, 7, 7]),
  (9, [2, 2, 2, 5, 8, 8, 8, 8]),
  (10, [1, 1, 1, 4]),
  (11, [0, 0, 0, 0])
]

-- Define a function to compute the frequencies of scores given the stem-and-leaf plot.
def score_frequencies (plot: List (Nat × List Nat)) : List (Nat × Nat) :=
  plot.foldl (fun acc (stem, leaves) =>
    leaves.foldl (fun acc leaf =>
      let score := 10 * stem + leaf;
      let idx := acc.findIdx (fun (s, _) => s = score);
      if idx = none then
        -- Add new score with frequency 1
        acc ++ [ (score, 1) ]
      else
        -- Increment frequency of existing score
        acc.mapIdx (fun i (s, f) => if i = idx.get else (s, f) else (s, f + 1))
    ) acc
  ) []

-- Define the mode as the score with the highest frequency.
def mode (freqs: List (Nat × Nat)) : Nat :=
  freqs.foldl (fun (acc: Nat × Nat) (score, count) =>
    if count > acc.snd then (score, count) else acc
  ) (0, 0).fst

-- Theaven or statement in Lean to prove that the mode is 98
theorem mode_is_98 : mode (score_frequencies stem_and_leaf_plot) = 98 :=
by sorry

end mode_is_98_l239_239708


namespace translate_one_chapter_in_three_hours_l239_239160

-- Definitions representing the conditions:
def jun_seok_time : ℝ := 4
def yoon_yeol_time : ℝ := 12

-- Question and Correct answer as a statement:
theorem translate_one_chapter_in_three_hours :
  (1 / (1 / jun_seok_time + 1 / yoon_yeol_time)) = 3 := by
sorry

end translate_one_chapter_in_three_hours_l239_239160


namespace complex_power_product_l239_239187

theorem complex_power_product (z : ℂ) (h : z = (1 - real.sqrt 3 * complex.I) / 2) :
    (∑ k in finset.filter (λ k: ℕ, k % 2 = 1) (finset.range 16), z ^ k) *
    (∑ k in finset.filter (λ k: ℕ, k % 2 = 1) (finset.range 16), (z ^ k)⁻¹) = 64 :=
by
  -- Placeholder for the proof
  sorry

end complex_power_product_l239_239187


namespace question_1_1_question_1_2_question_2_question_3_l239_239923

noncomputable def f (x : ℝ) : ℝ

axiom f_property : ∀ x : ℝ, f(x) + f(1 - x) = 2

theorem question_1_1 : f(1/2) = 1 := 
by
  sorry

theorem question_1_2 (n : ℕ) (h : n > 0) : f(1/n) + f((n-1)/n) = 2 := 
by
  sorry

def a_n (n : ℕ) (h : n > 0) : ℝ := f 0 + (∑ i in finset.range n, f (i / n)) + f 1

theorem question_2 (n : ℕ) (h : n > 0) : a_n n h = n + 1 :=
by
  sorry

def b_n (n : ℕ) (h : n > 0) : ℝ := 1 / (a_n n h - 1)
def S_n (n : ℕ) : ℝ := 4 * n / (2 * n + 1)
def T_n (n : ℕ) (h : n > 0) : ℝ := ∑ i in finset.range n, (b_n (i+1) (nat.succ_pos i))^2

theorem question_3 (n : ℕ) (h : n > 0) : T_n n h < S_n n :=
by
  sorry

end question_1_1_question_1_2_question_2_question_3_l239_239923


namespace meaning_of_implication_l239_239075

theorem meaning_of_implication (p q : Prop) : (p → q) = ((p → q) = True) :=
sorry

end meaning_of_implication_l239_239075


namespace smallest_solution_exists_l239_239052

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l239_239052


namespace hyperbola_focal_length_l239_239459

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) :
  (∀ x y : ℝ, (3 * x^2 - m^2 * y^2 = m^2) → (3 * x + m * y = 0)) → 
  (2 * real.sqrt (m + 1) = 4) :=
by 
  sorry

end hyperbola_focal_length_l239_239459


namespace mass_of_10_moles_is_10800_l239_239746

-- Define the mass of one mole of the compound
def molecularWeight : ℕ := 1080

-- Define the number of moles
def numberOfMoles : ℕ := 10

-- Define the total mass as a function of number of moles and molecular weight
def totalMass (n : ℕ) (mw : ℕ) : ℕ := n * mw

-- The theorem stating the mass of 10 moles of the compound given the molecular weight is 1080
theorem mass_of_10_moles_is_10800 : totalMass numberOfMoles molecularWeight = 10800 := by
  calc
    totalMass numberOfMoles molecularWeight = numberOfMoles * molecularWeight : rfl
    ... = 10 * 1080 : rfl
    ... = 10800 : by norm_num

end mass_of_10_moles_is_10800_l239_239746


namespace part_a_part_b_l239_239776

def problem_matrix (A : matrix (fin 24) (fin 25) ℕ) : Prop :=
  ∀ i : fin 25, ∃ j : fin 24, A j i = 1

theorem part_a (A : matrix (fin 24) (fin 25) ℕ) (hA : problem_matrix A) :
  ∃ x : fin 25 → ℕ, ∀ i : fin 24, ∑ j, A i j * x j % 2 = 0 :=
sorry

theorem part_b (A : matrix (fin 24) (fin 25) ℕ) (hA : problem_matrix A) :
  ∃ (x : fin 25 → ℤ), ∀ i : fin 24, ∑ j, A i j * x j = 0 :=
sorry

end part_a_part_b_l239_239776


namespace min_value_t_minus_2sqrt2_eq_3_l239_239909

theorem min_value_t_minus_2sqrt2_eq_3 {m n : ℝ} (hm : m > 0) (hn : n > 0) (h : 2 * m + n = m * n) :
  let t := min (m + n) in t - 2 * real.sqrt 2 = 3 :=
sorry

end min_value_t_minus_2sqrt2_eq_3_l239_239909


namespace hyperbola_focal_length_l239_239480

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239480


namespace determine_m_n_determine_monotonicity_intervals_determine_alpha_range_l239_239922

-- Given definitions and conditions
def f (x : ℝ) (m n : ℝ) : ℝ := m * x + n / x
def tangent_line (x y : ℝ) : Prop := 3 * x + y - 8 = 0
def tangent_point (a : ℝ) : Prop := tangent_line 1 a ∧ a = 5

theorem determine_m_n : ∃ (m n : ℝ), f 1 m n = 5 ∧ (m + n = 5) ∧ (m - n = -3) := 
by {
  use [1, 4],
  split, -- f(1) = 5
  sorry, -- proof of f(1) = 5
  split, -- m + n = 5 
  sorry, -- proof of m + n = 5
  -- m - n = -3
  sorry, -- proof of m - n = -3
}

theorem determine_monotonicity_intervals (m n : ℝ) (h1 : m = 1) (h2 : n = 4) :
  (∀ x < -2, deriv (λ x, f x m n) x > 0) ∧
  (∀ x, -2 < x ∧ x < 0, deriv (λ x, f x m n) x < 0) ∧
  (∀ x, 0 < x ∧ x < 2, deriv (λ x, f x m n) x < 0) ∧
  (∀ x > 2, deriv (λ x, f x m n) x > 0) :=
by {
  sorry -- proof of monotonicity intervals with m = 1 and n = 4
}

theorem determine_alpha_range (m n : ℝ) (h1 : m = 1) (h2 : n = 4) :
  { α : ℝ | α ∈ [0, Real.pi / 4) ∪ (Real.pi / 2, Real.pi) } :=
by {
  sorry -- proof of the range of α
}

end determine_m_n_determine_monotonicity_intervals_determine_alpha_range_l239_239922


namespace sailboat_speed_max_power_correct_l239_239685

noncomputable def sailboat_speed_max_power
  (B S ρ v_0 v : ℝ)
  (hS : S = 7)
  (hv0 : v_0 = 6.3)
  (F : ℝ → ℝ := λ v, (B * S * ρ * (v_0 - v) ^ 2) / 2)
  (N : ℝ → ℝ := λ v, F v * v) : Prop :=
  N v = (N (6.3 / 3)) ∧ v = 6.3 / 3

theorem sailboat_speed_max_power_correct :
  sailboat_speed_max_power B S ρ 6.3 2.1 :=
by
  -- Following the above methodology and conditions, we can verify
  -- that the speed v = 6.3 / 3 satisfies the condition when the power
  -- N(v') reaches its maximum value.
  sorry

end sailboat_speed_max_power_correct_l239_239685


namespace smallest_n_integer_l239_239001

noncomputable def x1 : ℝ := real.cbrt 4

noncomputable def x2 : ℝ := x1 ^ real.cbrt 4

noncomputable def x3 : ℝ := x2 ^ real.cbrt 4

noncomputable def x4 : ℝ := x3 ^ real.cbrt 4

theorem smallest_n_integer : ∃ n : ℕ, x n ∈ set.univ ∧ x n ≥ 1.0 ∧ x n % 1 = 0 :=
by
  use 4
  sorry

end smallest_n_integer_l239_239001


namespace plywood_width_is_5_l239_239800

theorem plywood_width_is_5 (length width perimeter : ℕ) (h1 : length = 6) (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 22) : width = 5 :=
by {
  -- proof steps would go here, but are omitted per instructions
  sorry
}

end plywood_width_is_5_l239_239800


namespace dice_probability_l239_239724

-- Define the set of outcomes for two six-sided dice
def outcomes : List (ℕ × ℕ) :=
  [ (i, j) | i ← List.range 1 7, j ← List.range 1 7 ]

-- Function to calculate the sum of the numbers on the two dice
def sum_outcome (x : ℕ × ℕ) : ℕ := x.fst + x.snd

-- Define the condition that the sum is at least 7 but less than 10
def condition (x : ℕ × ℕ) : Prop :=
  7 ≤ sum_outcome x ∧ sum_outcome x < 10

-- Calculate the probability by counting the satisfactory outcomes and dividing by the total outcomes (36)
theorem dice_probability :
  let favorable_outcomes := outcomes.filter condition
  (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ) = 5 / 12 :=
by
  sorry

end dice_probability_l239_239724


namespace part1_solution_part2_solution_l239_239552

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l239_239552


namespace solve_for_x_l239_239399

theorem solve_for_x (x : ℝ) (h : sqrt (1 - 4 * x) = 5) : x = -6 :=
by
  sorry

end solve_for_x_l239_239399


namespace find_largest_n_l239_239403

theorem find_largest_n (n : ℕ) (h_eq : (∑ i in Finset.range n.succ, Int.floor (Real.log i / Real.log 2)) = 1994) : n = 312 :=
sorry

end find_largest_n_l239_239403


namespace smallest_solution_l239_239047

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l239_239047


namespace smallest_solution_l239_239061

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239061


namespace function_intersection_le_one_l239_239644

theorem function_intersection_le_one (f : ℝ → ℝ)
  (h : ∀ x t : ℝ, t ≠ 0 → t * (f (x + t) - f x) > 0) :
  ∀ a : ℝ, ∃! x : ℝ, f x = a :=
by 
sorry

end function_intersection_le_one_l239_239644


namespace tangent_line_eq_extreme_values_interval_l239_239518

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l239_239518


namespace magnitude_w_eq_one_l239_239165

noncomputable def z := (complex.mk (-7) 15)^4 * (complex.mk 18 (-9))^3 / (complex.mk 5 2)
def w := z.conj / z

theorem magnitude_w_eq_one : complex.abs w = 1 := by
  sorry

end magnitude_w_eq_one_l239_239165


namespace find_side_length_l239_239142

theorem find_side_length
  (X : ℕ)
  (h1 : 3 + 2 + X + 4 = 12) :
  X = 3 :=
by
  sorry

end find_side_length_l239_239142


namespace focal_length_of_hyperbola_l239_239488

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239488


namespace find_q_l239_239300

theorem find_q (P J T : ℝ) (Q : ℝ) (q : ℚ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : T = P * (1 - Q))
  (h4 : Q = q / 100) :
  q = 6.25 := 
by
  sorry

end find_q_l239_239300


namespace inverse_comp_inverse_comp_inverse_g4_l239_239251

noncomputable def g : ℕ → ℕ
| 1 := 4
| 2 := 5
| 3 := 1
| 4 := 3
| 5 := 2
| _ := 0 -- not needed, just for completeness

theorem inverse_comp_inverse_comp_inverse_g4 : function.inv_fun g (function.inv_fun g (function.inv_fun g 4)) = 4 :=
by sorry

end inverse_comp_inverse_comp_inverse_g4_l239_239251


namespace shortest_tangent_length_l239_239177

-- Definitions for the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Proof statement for the length of the shortest tangent line segment between C1 and C2
theorem shortest_tangent_length :
  let P := (12, 0),
      Q := (-18, 0),
      r1 := 7,
      r2 := 8,
      d := 30 in
  sqrt (207 : ℝ) + sqrt (132 : ℝ)

end shortest_tangent_length_l239_239177


namespace star_assoc_l239_239186

noncomputable def alpha : ℝ := Classical.some (exists_pos_root_of_cubic_ne_zero 1 (-1991) 1 0)

axiom alpha_property : alpha^2 = 1991 * alpha + 1

def star (m n : ℕ) : ℕ := 
  m * n + ⌊alpha * m⌋ * ⌊alpha * n⌋

theorem star_assoc (p q r : ℕ) : 
  star (star p q) r = star p (star q r) := 
by sorry

end star_assoc_l239_239186


namespace sin_double_angle_in_fourth_quadrant_l239_239944

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239944


namespace smallest_of_five_consecutive_numbers_l239_239284

theorem smallest_of_five_consecutive_numbers (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → 
  n = 18 :=
by sorry

end smallest_of_five_consecutive_numbers_l239_239284


namespace set_B_correct_l239_239432

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l239_239432


namespace opponent_score_l239_239138

theorem opponent_score (s g c total opponent : ℕ)
  (h1 : s = 20)
  (h2 : g = 2 * s)
  (h3 : c = 2 * g)
  (h4 : total = s + g + c)
  (h5 : total - 55 = opponent) :
  opponent = 85 := by
  sorry

end opponent_score_l239_239138


namespace sin_double_angle_in_fourth_quadrant_l239_239942

theorem sin_double_angle_in_fourth_quadrant (α : ℝ) (h : -π/2 < α ∧ α < 0) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_in_fourth_quadrant_l239_239942


namespace solve_equation_1_solve_equation_2_l239_239221

theorem solve_equation_1 (x : ℝ) : x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6 := 
by sorry

end solve_equation_1_solve_equation_2_l239_239221


namespace sum_of_elements_in_T_l239_239173

-- Definition of the set T
def T : Set ℕ := {x | 16 ≤ x ∧ x ≤ 31}

-- Sum of all elements in T
theorem sum_of_elements_in_T : ∑ x in T, x = 248 :=
by
  sorry

end sum_of_elements_in_T_l239_239173


namespace fraction_meaningless_l239_239713

theorem fraction_meaningless (a b : ℤ) (h1 : a = 4) (h2 : b = -4) : (3 : ℚ) / (a + b) = 0 :=
by
  have hab : a + b = 0 := by
    rw [h1, h2]
    norm_num
  rw [hab]
  apply div_zero
  sorry

end fraction_meaningless_l239_239713


namespace exists_disjoint_arithmetic_partitions_l239_239159

theorem exists_disjoint_arithmetic_partitions :
  ∃ n : ℕ, ∃ (A B : finset ℕ),
  (∀ (x ∈ A) (y ∈ B), x ≠ y) ∧
  (∀ k (k ∈ A), k ∈ finset.range (n + 1)) ∧
  (∀ k (k ∈ B), k ∈ finset.range (n + 1)) ∧
  (∃ d1 : ℕ, d1 ≠ 0 ∧
  ∃ a1 : ℕ, A = finset.range (d1 * ⌊(n - a1) / d1⌋ + 1) \ {finset.range a1}) ∧
  (∃ d2 : ℕ, d2 ≠ 0 ∧ d2 ≠ d1 ∧
  ∃ a2 : ℕ, B = finset.range (d2 * ⌊(n - a2) / d2⌋ + 1) \ {finset.range a2}) ∧
  (3 ≤ A.card) ∧
  (3 ≤ B.card) := 
sorry

end exists_disjoint_arithmetic_partitions_l239_239159


namespace freq_distribution_correct_l239_239882

variable (freqTable_isForm : Prop)
variable (freqHistogram_isForm : Prop)
variable (freqTable_isAccurate : Prop)
variable (freqHistogram_isIntuitive : Prop)

theorem freq_distribution_correct :
  ((freqTable_isForm ∧ freqHistogram_isForm) ∧
   (freqTable_isAccurate ∧ freqHistogram_isIntuitive)) →
  True :=
by
  intros _
  exact trivial

end freq_distribution_correct_l239_239882


namespace table_sums_congruence_mod_n4_l239_239626

open BigOperators

theorem table_sums_congruence_mod_n4 (n : ℕ) (hn : n > 1) 
  (table : Fin n → Fin n → ℤ) 
  (h_cell : ∀ i j, table i j % n = 1)
  (h_row_sum : ∀ i, (∑ j, table i j) % (n ^ 2) = n)
  (h_col_sum : ∀ j, (∑ i, table i j) % (n ^ 2) = n) :
  ((Finset.univ.sum (λ i, (Finset.univ.prod (λ j, table i j)))) % n ^ 4) = 
  ((Finset.univ.sum (λ j, (Finset.univ.prod (λ i, table i j)))) % n ^ 4) := sorry

end table_sums_congruence_mod_n4_l239_239626


namespace find_perimeter_of_EFGH_l239_239151

variable (EF FG GH : ℝ)
variables (right_angle_F : ∠(Line.mk EF FG) = 90)
variable (perp_EH_HG : ⊤.perp (Line.mk EH HG))
variable (EF_len : EF = 15)
variable (FG_len : FG = 20)
variable (GH_len : GH = 9)

theorem find_perimeter_of_EFGH
  (EF FG GH : ℝ)
  (right_angle_F : ∠(Line.mk EF FG) = 90)
  (perp_EH_HG : ⊤.perp (Line.mk EH HG))
  (EF_len : EF = 15)
  (FG_len : FG = 20)
  (GH_len : GH = 9) :
  (∃ EH, EH = sqrt (25^2 + 9^2)) ∧
  (∃ P, P = EF + FG + GH + sqrt 706) := by
  sorry

end find_perimeter_of_EFGH_l239_239151


namespace second_smallest_packs_of_hot_dogs_l239_239848

theorem second_smallest_packs_of_hot_dogs (n m : ℕ) (k : ℕ) :
  (12 * n ≡ 5 [MOD 10]) ∧ (10 * m ≡ 3 [MOD 12]) → n = 15 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l239_239848


namespace correct_function_is_option_2_l239_239336

def options (n : Nat) : (ℝ → ℝ) :=
  match n with
  | 1 => fun x => x^3
  | 2 => fun x => abs x + 1
  | 3 => fun x => -x^2 + 1
  | 4 => fun x => 2^(-x)
  | _ => fun x => 0  -- a default, should not be used

theorem correct_function_is_option_2 :
  (∀ x : ℝ, x ≥ 0 → options 2 (x + 1) > options 2 x) ∧ (∀ x : ℝ, options 2 x = options 2 (-x)) :=
by 
  sorry

end correct_function_is_option_2_l239_239336


namespace find_coordinates_of_Q_l239_239153

/-- Point P in the Cartesian coordinate system -/
def P : ℝ × ℝ := (1, 2)

/-- Q is below the x-axis -/
def Q {y : ℝ} : ℝ × ℝ := (1, y)

/-- Distance between P and Q is 5 and Q is below x-axis -/
theorem find_coordinates_of_Q (y : ℝ) (h1 : y < 0) (h2 : (P.2 - y) = 5) : Q = (1, -3) :=
sorry

end find_coordinates_of_Q_l239_239153


namespace count_four_digit_numbers_without_1_or_4_l239_239936

-- Define a function to check if a digit is allowed (i.e., not 1 or 4)
def allowed_digit (d : ℕ) : Prop := d ≠ 1 ∧ d ≠ 4

-- Function to count four-digit numbers without digits 1 or 4
def count_valid_four_digit_numbers : ℕ :=
  let valid_first_digits := [2, 3, 5, 6, 7, 8, 9]
  let valid_other_digits := [0, 2, 3, 5, 6, 7, 8, 9]
  (valid_first_digits.length) * (valid_other_digits.length ^ 3)

-- The main theorem stating that the number of valid four-digit integers is 3072
theorem count_four_digit_numbers_without_1_or_4 : count_valid_four_digit_numbers = 3072 :=
by
  sorry

end count_four_digit_numbers_without_1_or_4_l239_239936


namespace triangle_area_perpendicular_intercepts_l239_239726

theorem triangle_area_perpendicular_intercepts
  (A : ℝ × ℝ)
  (b1 b2 : ℝ)
  (hA : A = (8, 6))
  (h_diff : b1 - b2 = 14)
  (h_perp : ∀ m1 m2, m1 * m2 = -1 → let y1 := λ x, m1 * x + b1 in
                                     let y2 := λ x, m2 * x + b2 in
                                     y1 8 = 6 ∧ y2 8 = 6) :
  ∃ P Q : ℝ × ℝ, P = (0, b1) ∧ Q = (0, b2) ∧ 
    1 / 2 * (P.2 - Q.2) * (A.1 - 0) = 56 :=
by
  sorry

end triangle_area_perpendicular_intercepts_l239_239726


namespace problem1_problem2_l239_239360

-- Statement for Problem 1
theorem problem1 (x y : ℝ) : (x - y) ^ 2 + x * (x + 2 * y) = 2 * x ^ 2 + y ^ 2 :=
by sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3 * x + 4) / (x - 1) + x) / ((x - 2) / (x ^ 2 - x)) = x ^ 2 - 2 * x :=
by sorry

end problem1_problem2_l239_239360


namespace value_of_a_l239_239281

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_a (a : ℝ) :
  (1 / log_base 2 a) + (1 / log_base 3 a) + (1 / log_base 4 a) + (1 / log_base 5 a) = 7 / 4 ↔
  a = 120 ^ (4 / 7) :=
by
  sorry

end value_of_a_l239_239281


namespace hyperbola_focal_length_l239_239498

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239498


namespace find_power_function_l239_239693

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(x - 4) + 1

def fixed_point (a : ℝ) : Prop := 
  (a > 0 ∧ a ≠ 1) ∧ g(a)(4) = 2

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, f(x) = x^b

theorem find_power_function (a : ℝ) (f : ℝ → ℝ) : 
  fixed_point(a) ∧ is_power_function(f)
  → f = (λ x, real.sqrt x) :=
by
  sorry

end find_power_function_l239_239693


namespace smallest_vertical_distance_between_graphs_l239_239774

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem smallest_vertical_distance_between_graphs :
  ∃ (d : ℝ), (∀ (x : ℝ), |f x - g x| ≥ d) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), |f x - g x| < d + ε) ∧ d = 3 / 4 :=
by
  sorry

end smallest_vertical_distance_between_graphs_l239_239774


namespace greatest_divisor_420_smaller_than_50_and_factor_of_90_l239_239729

theorem greatest_divisor_420_smaller_than_50_and_factor_of_90 : 
  ∃ d, d ∣ 420 ∧ d ∣ 90 ∧ d < 50 ∧ ∀ k, k ∣ 420 ∧ k ∣ 90 ∧ k < 50 → k ≤ d := 
begin
  use 30,
  split,
  { exact dvd_refl 420 },     -- 30 is a divisor of 420
  split,
  { exact dvd_refl 90 },      -- 30 is a divisor of 90
  split,
  { linarith },               -- 30 < 50
  intros k hk,
  cases hk with hk1 hk_rest,
  cases hk_rest with hk2 hk_lt,
  sorry                     -- missing the internal proof steps
end

end greatest_divisor_420_smaller_than_50_and_factor_of_90_l239_239729


namespace prob_volunteer_A_not_assigned_to_A_l239_239880

theorem prob_volunteer_A_not_assigned_to_A : 
  (∀ volunteer community, volunteer ∈ {1, 2, 3, 4} → community ∈ {1, 2, 3, 4} → volunteer ≠ community) →
  ((3 / 4) = 3 / 4) := 
begin
  sorry
end

end prob_volunteer_A_not_assigned_to_A_l239_239880


namespace find_common_difference_l239_239630

-- Let a_n be the arithmetic sequence
-- Define the first term and the sum of the first three terms
def a (n : ℕ) : ℤ :=
  if n = 1 then -2 else -2 + (n - 1) * d

def S (n : ℕ) : ℤ :=
  n * (-2) + (n * (n - 1) / 2) * d

theorem find_common_difference (d : ℤ) :
  (a 1 = -2) ∧ (S 3 = 0) → d = 2 :=
by
  intro h
  -- Insert actual proof steps here
  sorry

end find_common_difference_l239_239630


namespace gum_pack_size_is_5_l239_239571
noncomputable def find_gum_pack_size (x : ℕ) : Prop :=
  let cherry_initial := 25
  let grape_initial := 40
  let cherry_lost := cherry_initial - 2 * x
  let grape_found := grape_initial + 4 * x
  (cherry_lost * grape_found) = (cherry_initial * grape_initial)

theorem gum_pack_size_is_5 : find_gum_pack_size 5 :=
by
  sorry

end gum_pack_size_is_5_l239_239571


namespace sin_double_angle_neg_l239_239946

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239946


namespace smallest_solution_to_equation_l239_239032

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239032


namespace jane_total_earnings_eq_122_45_l239_239162

noncomputable def total_earnings : ℝ :=
  let tulip_bulbs := 20
  let tulip_earnings := tulip_bulbs * 0.50
  let iris_bulbs := tulip_bulbs / 2
  let iris_earnings := iris_bulbs * 0.40
  let hyacinth_bulbs := iris_bulbs + iris_bulbs / 3
  let hyacinth_earnings := hyacinth_bulbs * 0.75
  let daffodil_bulbs := 30
  let daffodil_earnings := daffodil_bulbs * 0.25
  let crocus_bulbs := daffodil_bulbs * 3
  let crocus_earnings := crocus_bulbs * 0.60
  let diff_bulbs := crocus_bulbs - daffodil_bulbs
  let gladiolus_bulbs := 2 * diff_bulbs + (daffodil_bulbs * 0.15).toInt
  let gladiolus_earnings := gladiolus_bulbs * 0.30
  tulip_earnings + iris_earnings + hyacinth_earnings + daffodil_earnings + crocus_earnings + gladiolus_earnings

theorem jane_total_earnings_eq_122_45 : total_earnings = 122.45 :=
  by sorry

end jane_total_earnings_eq_122_45_l239_239162


namespace rational_terms_and_largest_coefficient_l239_239990

variables (x : ℝ)

def binom (n k : ℕ) := Nat.choose n k

theorem rational_terms_and_largest_coefficient :
  (∀ n : ℕ, (∀ (a1 a2 a3 : ℝ), a1 = 1 ∧ a2 = 1/2 * binom n 1 ∧ a3 = 1/4 * binom n 2 → 2 * a2 = a1 + a3 → n = 8) →
  (∀ (r : ℕ), r = 0 ∨ r = 4 ∨ r = 8 →
    (∀ (k : ℕ), k = r + 1 → 
      (T_x : ℝ) = binom 8 r * (sqrt x)^(8 - r) * (1/(2 * sqrt (4 * x)))^r =
        (1/2)^r * binom 8 r * x^((16 - 3 * r)/4) → 
        T_x = x^4 ∨ T_x = (35/8) * x ∨ T_x = (1/(256 * x^2))) →
  (∀ (r : ℕ), (r = 2 ∨ r = 3) →
    (T_largest : ℝ) = binom 8 r * (sqrt x)^(8 - r) * (1/(2 * sqrt (4 * x)))^r →
      T_largest = 7 * x^(5/2) ∨ T_largest = 7 * x^(7/4))) 
by
  sorry

end rational_terms_and_largest_coefficient_l239_239990


namespace reflection_of_A_across_G_is_correct_l239_239721

-- Vertices of the triangle
def A : ℂ := 1 + I
def B : ℂ := -3 - I
def C : ℂ := 2 - 3 * I

-- Centroid of the triangle
def G : ℂ := (A + B + C) / 3

-- Reflection of point A across point G 
def reflection (P Q : ℂ) : ℂ := 2 * Q - P

-- Statement to prove
theorem reflection_of_A_across_G_is_correct :
  reflection A G = -1 - 3 * I :=
by {
  sorry
}

end reflection_of_A_across_G_is_correct_l239_239721


namespace sector_perimeter_l239_239425

-- Conditions:
def theta : ℝ := 54  -- central angle in degrees
def r : ℝ := 20      -- radius in cm

-- Translation of given conditions and expected result:
theorem sector_perimeter (theta_eq : theta = 54) (r_eq : r = 20) :
  let l := (θ * r) / 180 * Real.pi 
  let perim := l + 2 * r 
  perim = 6 * Real.pi + 40 := sorry

end sector_perimeter_l239_239425


namespace combined_build_time_l239_239020

noncomputable def combined_time_in_years (F E : ℕ) (h1 : F = 30) (h2 : F * 2 = E) : ℕ :=
(F + E) / 12

theorem combined_build_time {F E : ℕ} (h1 : F = 30) (h2 : F * 2 = E) :
  combined_time_in_years F E h1 h2 = 7.5 :=
by
  sorry

end combined_build_time_l239_239020


namespace least_whole_number_subtracted_l239_239302

theorem least_whole_number_subtracted (x : ℕ) :
  ((6 - x) / (7 - x) < (16 / 21)) → x = 3 :=
by
  sorry

end least_whole_number_subtracted_l239_239302


namespace angle_of_inclination_l239_239245

-- Define the given conditions and the target statement in Lean 4
theorem angle_of_inclination (a m : ℝ) (m_nonzero : m ≠ 0) (passes_through_point : a + m - 2 * a = 0) : 
  is_angle_of_inclination (ax + my - 2a = 0) (135 : ℝ) :=
sorry

end angle_of_inclination_l239_239245


namespace solution_set_empty_l239_239073

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l239_239073


namespace leo_weight_l239_239985

variable (L K : ℝ)

theorem leo_weight (h1 : L + 12 = 1.7 * K)
                   (h2 : L + K = 210) :
                   L ≈ 127.78 :=
by
  sorry

end leo_weight_l239_239985


namespace expression_of_f_volume_of_solid_of_revolution_l239_239886

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x + 1

def g (x : ℝ) : ℝ := x * f x

-- Proof problems
theorem expression_of_f :
  ∀ x, f x = -2 * x + 1 := 
sorry

theorem volume_of_solid_of_revolution :
  ∫ x in 0..(1 / 2:ℝ), (g x)^2 = π / 240 := 
sorry

end expression_of_f_volume_of_solid_of_revolution_l239_239886


namespace find_radius_l239_239158

theorem find_radius
  (r_1 r_2 r_3 : ℝ)
  (h_cone : r_2 = 2 * r_1 ∧ r_3 = 3 * r_1 ∧ r_1 + r_2 + r_3 = 18) :
  r_1 = 3 :=
by
  sorry

end find_radius_l239_239158


namespace find_F_l239_239986

theorem find_F (F C : ℝ) (h1 : C = 4/7 * (F - 40)) (h2 : C = 28) : F = 89 := 
by
  sorry

end find_F_l239_239986


namespace f_maps_S_to_S_exists_t_smallest_t_l239_239179

open Nat Int

-- Define S and f based on the problem statement.
def S : Set (ℕ × ℕ) := {p | coprime p.1 p.2 ∧ p.2 % 2 = 0 ∧ p.1 < p.2}

def f (s : ℕ × ℕ) : ℕ × ℕ :=
  let (m, n) := s
  let (k, n₀) := (log2(n / lowOddFactor n), lowOddFactor n)
  (n₀, m + n - n₀)

-- Define helper functions
def lowOddFactor : ℕ → ℕ
  | 0 => 1
  | n => if k % 2 = 0 then (n / 2^k) else lowOddFactor (n / 2^k)
      where k : ℕ := n.trailingZeroCount

-- To express the fixed point t
def iter (f : α → α) (t : ℕ) (s : α) : α :=
  match t with
  | 0 => s
  | succ t' => f (iter f t' s)

-- Main theorem statements
theorem f_maps_S_to_S (s : ℕ × ℕ) (hs : s ∈ S) : f s ∈ S := by
  sorry

theorem exists_t (s : ℕ × ℕ) (hs : s ∈ S) : ∃ t : ℕ, 1 ≤ t ∧ t ≤ (s.1 + s.2 + 1) / 4 ∧ iter f t s = s := by
  sorry

theorem smallest_t (s : ℕ × ℕ) (hs : s ∈ S) (hs_prime : isPrime (s.1 + s.2)) (coprime_cond : ∀ k, 1 ≤ k → k ≤ s.1 + s.2 - 2 → ¬((s.1 + s.2) ∣ (2^k - 1))) :
  ∃ t : ℕ, t = (s.1 + s.2 + 1) / 4 ∧ iter f t s = s := by
  sorry

end f_maps_S_to_S_exists_t_smallest_t_l239_239179


namespace sin_double_angle_neg_of_fourth_quadrant_l239_239964

variable (α : ℝ)

def is_in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -π / 2 + 2 * k * π < α ∧ α < 2 * k * π

theorem sin_double_angle_neg_of_fourth_quadrant (h : is_in_fourth_quadrant α) : sin (2 * α) < 0 :=
sorry

end sin_double_angle_neg_of_fourth_quadrant_l239_239964


namespace total_pages_in_science_fiction_section_l239_239696

def number_of_books : ℕ := 8
def pages_per_book : ℕ := 478

theorem total_pages_in_science_fiction_section : number_of_books * pages_per_book = 3824 :=
by simp [number_of_books, pages_per_book]; sorry

end total_pages_in_science_fiction_section_l239_239696


namespace polynomial_form_l239_239389

def homogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def cyclic_sum_zero (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, P (a+b) c + P (b+c) a + P (c+a) b = 0

noncomputable def specific_value (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem polynomial_form :
  ∀ (P : ℝ → ℝ → ℝ) (n : ℕ),
  (homogeneous P n) →
  (cyclic_sum_zero P) →
  (specific_value P) →
  (∀ x y : ℝ, P x y = (x + y)^(n-1) * (x - 2y)) :=
by
  sorry

end polynomial_form_l239_239389


namespace smallest_solution_l239_239038

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239038


namespace cuberoot_eq_neg4_has_solution_l239_239859

theorem cuberoot_eq_neg4_has_solution (x : ℝ) : (∃ x : ℝ, x = 222 ∧ (∛ (10 - x / 3)) = -4) :=
sorry

end cuberoot_eq_neg4_has_solution_l239_239859


namespace min_value_quadratic_expression_l239_239864

theorem min_value_quadratic_expression :
  ∃ x y : ℝ, min_val (3*x^2 + 3*x*y + y^2 - 3*x + 3*y + 9) = (45 / 8) := 
sorry

end min_value_quadratic_expression_l239_239864


namespace setB_is_correct_l239_239434

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l239_239434


namespace sam_correct_percent_l239_239597

variable (y : ℝ)
variable (h_pos : 0 < y)

theorem sam_correct_percent :
  ((8 * y - 3 * y) / (8 * y) * 100) = 62.5 := by
sorry

end sam_correct_percent_l239_239597


namespace part1_solution_set_part2_range_of_a_l239_239535

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239535


namespace part1_part2_l239_239544

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l239_239544


namespace complete_square_expression_l239_239590

theorem complete_square_expression :
  ∃ (a h k : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  sorry

end complete_square_expression_l239_239590


namespace total_revenue_is_correct_l239_239718

def category_a_price : ℝ := 65
def category_b_price : ℝ := 45
def category_c_price : ℝ := 25

def category_a_discounted_price : ℝ := category_a_price - 0.55 * category_a_price
def category_b_discounted_price : ℝ := category_b_price - 0.35 * category_b_price
def category_c_discounted_price : ℝ := category_c_price - 0.20 * category_c_price

def category_a_full_price_quantity : ℕ := 100
def category_b_full_price_quantity : ℕ := 50
def category_c_full_price_quantity : ℕ := 60

def category_a_discounted_quantity : ℕ := 20
def category_b_discounted_quantity : ℕ := 30
def category_c_discounted_quantity : ℕ := 40

def revenue_from_category_a : ℝ :=
  category_a_discounted_quantity * category_a_discounted_price +
  category_a_full_price_quantity * category_a_price

def revenue_from_category_b : ℝ :=
  category_b_discounted_quantity * category_b_discounted_price +
  category_b_full_price_quantity * category_b_price

def revenue_from_category_c : ℝ :=
  category_c_discounted_quantity * category_c_discounted_price +
  category_c_full_price_quantity * category_c_price

def total_revenue : ℝ :=
  revenue_from_category_a + revenue_from_category_b + revenue_from_category_c

theorem total_revenue_is_correct :
  total_revenue = 12512.50 :=
by
  unfold total_revenue
  unfold revenue_from_category_a
  unfold revenue_from_category_b
  unfold revenue_from_category_c
  unfold category_a_discounted_price
  unfold category_b_discounted_price
  unfold category_c_discounted_price
  sorry

end total_revenue_is_correct_l239_239718


namespace sam_sandwiches_count_l239_239806

theorem sam_sandwiches_count :
  let total_combinations := 5 * 7 * 6
  let beef_mozzarella_invalid := 5
  let rye_turkey_invalid := 6
  let turkey_mozzarella_invalid := 5
  total_combinations - beef_mozzarella_invalid - rye_turkey_invalid - turkey_mozzarella_invalid = 194 :=
by {
  let total_combinations := 5 * 7 * 6;
  let beef_mozzarella_invalid := 5;
  let rye_turkey_invalid := 6;
  let turkey_mozzarella_invalid := 5;
  have total_invalid := beef_mozzarella_invalid + rye_turkey_invalid + turkey_mozzarella_invalid;
  have final_count := total_combinations - total_invalid;
  exact final_count = 194
}

end sam_sandwiches_count_l239_239806


namespace simplify_poly_eq_l239_239084

-- Variables declaration
variables (x y : ℝ)

-- Definition of y in terms of x
def y_def : Prop := y = x + 1 / x

-- Initial polynomial equation
def poly_eq : Prop := x^4 + x^3 - 7x^2 + x + 1 = 0

-- Resultant polynomial equation in y
def resultant_poly_eq : Prop := x^2 * (y^2 + y - 9) = 0

-- The proof statement
theorem simplify_poly_eq (h1 : y_def x y) (h2 : poly_eq x) : resultant_poly_eq x y := 
by
  sorry

end simplify_poly_eq_l239_239084


namespace bus_length_is_200_l239_239763

def length_of_bus (distance_km distance_secs passing_secs : ℕ) : ℕ :=
  let speed_kms := distance_km / distance_secs
  let speed_ms := speed_kms * 1000
  speed_ms * passing_secs

theorem bus_length_is_200 
  (distance_km : ℕ) (distance_secs : ℕ) (passing_secs : ℕ)
  (h1 : distance_km = 12) (h2 : distance_secs = 300) (h3 : passing_secs = 5) : 
  length_of_bus distance_km distance_secs passing_secs = 200 := 
  by
    sorry

end bus_length_is_200_l239_239763


namespace part1_solution_set_part2_range_of_a_l239_239540

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l239_239540


namespace hyperbola_focal_length_l239_239493

theorem hyperbola_focal_length (m : ℝ) (h : m > 0) (asymptote : ∀ x y : ℝ, sqrt 3 * x + m * y = 0) :
  let C := { p : ℝ × ℝ | (p.1^2 / m - p.2^2 = 1) } in
  let focal_length : ℝ := 4 in
  True := by
  sorry

end hyperbola_focal_length_l239_239493


namespace ab_sum_l239_239108

def f (x : ℝ) : ℝ := 3 * x + x - 5

theorem ab_sum (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b - a = 1) (x0 : ℝ) (h4 : x0 ∈ set.Icc a b) :
  a + b = 3 :=
by
  sorry

end ab_sum_l239_239108


namespace determine_a_l239_239691

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem determine_a :
  {a : ℝ | 0 < a ∧ (f (a + 1) ≤ f (2 * a^2))} = {a : ℝ | 1 ≤ a ∧ a ≤ Real.sqrt 6 / 2 } :=
by
  sorry

end determine_a_l239_239691


namespace andrew_total_appeizers_count_l239_239821

theorem andrew_total_appeizers_count :
  let hotdogs := 30
  let cheese_pops := 20
  let chicken_nuggets := 40
  hotdogs + cheese_pops + chicken_nuggets = 90 := 
by 
  sorry

end andrew_total_appeizers_count_l239_239821


namespace smallest_solution_l239_239065

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l239_239065


namespace correct_propositions_l239_239305

-- Definitions of geometric relationships
def Line := Type
def Plane := Type

def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def lies_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_plane (p1 p2 : Plane) : Prop := sorry
def angle_eq (l1 l2 : Line) (p1 p2 : Plane) : Prop := sorry

-- Propositions from the problem

-- Proposition B: Given conditions and prove m is perpendicular to n
def Proposition_B (m n : Line) (α : Plane) :=
  perpendicular_plane m α → parallel n α → perpendicular m n

-- Proposition C: Given conditions and prove m is parallel to β
def Proposition_C (m : Line) (α β : Plane) :=
  parallel_plane α β → lies_in m α → parallel m β

-- Proposition D: Given conditions and prove angles formed by m and α are equal to angles formed by n and β
def Proposition_D (m n : Line) (α β : Plane) :=
  parallel m n → parallel_plane α β → angle_eq m n α β

-- Combined problem statement to prove propositions B, C, and D
theorem correct_propositions (m n : Line) (α β : Plane) :
  Proposition_B m n α ∧ Proposition_C m α β ∧ Proposition_D m n α β :=
by
  split
  · sorry
  split
  · sorry
  · sorry

end correct_propositions_l239_239305


namespace length_of_EF_l239_239605

theorem length_of_EF (AB BC : ℝ) (DE DF : ℝ) (Area_ABC : ℝ) (Area_DEF : ℝ) (EF : ℝ) 
  (h₁ : AB = 10) (h₂ : BC = 15) (h₃ : DE = DF) (h₄ : Area_DEF = (1/3) * Area_ABC) 
  (h₅ : Area_ABC = AB * BC) (h₆ : Area_DEF = (1/2) * (DE * DF)) : 
  EF = 10 * Real.sqrt 2 := 
by 
  sorry

end length_of_EF_l239_239605


namespace max_vertices_no_rectangle_l239_239740

/-- In a regular 2016-sided polygon (2016-gon), prove that the maximum number of vertices 
    that can be marked such that no four marked vertices form the vertices of a rectangle is 1009. -/
theorem max_vertices_no_rectangle (n : ℕ) (h : n = 2016) : 
  ∃ (m : ℕ), m = 1009 ∧ 
    ∀ (marked : finset (fin n)), 
      marked.card ≤ m → 
      (¬ ∃ (a b c d : fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
        (is_rectangle a b c d)) := 
begin 
  sorry 
end

/-- Predicate to determine if four vertices form a rectangle in a regular n-gon. -/
def is_rectangle (a b c d : fin 2016) : Prop := 
  ∃ (k : ℕ), k ∈ finset.range 1008 ∧ 
    ((a = fin.of_nat k) ∧ (b = fin.of_nat (k + 1008)) ∧ 
     (c = fin.of_nat (k + 1008 + 1)) ∧ (d = fin.of_nat (k + 1)) ∨ 
     (a = fin.of_nat (k + 1008)) ∧ (b = fin.of_nat k) ∧ 
     (c = fin.of_nat (k + 1)) ∧ (d = fin.of_nat (k + 1008 + 1)))

end max_vertices_no_rectangle_l239_239740


namespace expected_voters_percentage_l239_239593

theorem expected_voters_percentage :
  let total_voters := 100
  let dem_percentage := 0.60
  let rep_percentage := 0.40
  let dem_for_A_percentage := 0.75
  let rep_for_A_percentage := 0.20
  let num_dem := dem_percentage * total_voters
  let num_rep := rep_percentage * total_voters
  let dem_for_A := dem_for_A_percentage * num_dem
  let rep_for_A := rep_for_A_percentage * num_rep
  let total_for_A := dem_for_A + rep_for_A
  let percentage_for_A := (total_for_A / total_voters) * 100
  in
  percentage_for_A = 53 :=
by
  -- Proof required
  sorry

end expected_voters_percentage_l239_239593


namespace find_a_l239_239401

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a (a : ℝ) : 
  (∀ x ∈ set.Icc (-3 : ℝ) 2, f a x ≤ 4) →
  (∃ x ∈ set.Icc (-3 : ℝ) 2, f a x = 4) →
  (a = -3 ∨ a = 3/8) :=
begin
  sorry
end

end find_a_l239_239401


namespace find_point_B_l239_239628

noncomputable def point_A : ℝ × ℝ := (2, 4)

def parabola (x : ℝ) : ℝ := x^2

def tangent_slope (x : ℝ) : ℝ := 2 * x

def normal_slope (x : ℝ) : ℝ := -1 / (tangent_slope x)

def normal_line (x : ℝ) : ℝ × ℝ → ℝ := 
  λ (p : ℝ × ℝ), 
  p.2 + normal_slope p.1 * (x - p.1)

theorem find_point_B :
  let A := (2 : ℝ, 4 : ℝ),
      B := (-9/4 : ℝ, 81/16 : ℝ) in
      normal_line (-9/4) A = parabola (-9/4) → 
      B = (-9/4 : ℝ, 81/16 : ℝ) := 
by
  intros A B h
  sorry

end find_point_B_l239_239628


namespace find_p_plus_q_l239_239260

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem find_p_plus_q :
  let xy := 15
  let yz := 20
  let xz := 13
  let area_triangle := area_of_triangle xy yz xz
  let b := 9 / 25
  1 / 2 * area_triangle = 36 →
  ∃ p q : ℕ, p + q = 34 ∧ b = p / q ∧ Nat.gcd p q = 1 :=
by
  have half_area := area_of_triangle 15 20 13 / 2
  have b := 9 / 25
  sorry

end find_p_plus_q_l239_239260


namespace farm_field_area_l239_239126

theorem farm_field_area
  (plough_per_day_planned plough_per_day_actual fields_left : ℕ)
  (D : ℕ) 
  (condition1 : plough_per_day_planned = 100)
  (condition2 : plough_per_day_actual = 85)
  (condition3 : fields_left = 40)
  (additional_days : ℕ) 
  (condition4 : additional_days = 2)
  (initial_days : D + additional_days = 85 * (D + 2) + 40) :
  (100 * D + fields_left = 1440) :=
by
  sorry

end farm_field_area_l239_239126


namespace hyperbola_focal_length_proof_l239_239454

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239454


namespace johns_cocktail_not_stronger_l239_239210

-- Define the percentage of alcohol in each beverage
def beer_percent_alcohol : ℝ := 0.05
def liqueur_percent_alcohol : ℝ := 0.10
def vodka_percent_alcohol : ℝ := 0.40
def whiskey_percent_alcohol : ℝ := 0.50

-- Define the weights of the liquids used in the cocktails
def john_liqueur_weight : ℝ := 400
def john_whiskey_weight : ℝ := 100
def ivan_vodka_weight : ℝ := 400
def ivan_beer_weight : ℝ := 100

-- Calculate the alcohol contents in each cocktail
def johns_cocktail_alcohol : ℝ := john_liqueur_weight * liqueur_percent_alcohol + john_whiskey_weight * whiskey_percent_alcohol
def ivans_cocktail_alcohol : ℝ := ivan_vodka_weight * vodka_percent_alcohol + ivan_beer_weight * beer_percent_alcohol

-- The proof statement to prove that John's cocktail is not stronger than Ivan's cocktail
theorem johns_cocktail_not_stronger :
  johns_cocktail_alcohol ≤ ivans_cocktail_alcohol :=
by
  -- Add proof steps here
  sorry

end johns_cocktail_not_stronger_l239_239210


namespace min_value_of_expression_l239_239876

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (u v : ℝ), (0 < u) → (0 < v) → (1/u + 1/v = 1) → (1/(u - 1) + 4/(v - 1)) ≥ m :=
begin
  use 4,
  split,
  { refl },
  { intros u v hu hv huv,
    sorry }
end

end min_value_of_expression_l239_239876


namespace ms_smith_books_divided_l239_239252

theorem ms_smith_books_divided (books_for_girls : ℕ) (girls boys : ℕ) (books_per_girl : ℕ)
  (h1 : books_for_girls = 225)
  (h2 : girls = 15)
  (h3 : boys = 10)
  (h4 : books_for_girls / girls = books_per_girl)
  (h5 : books_per_girl * boys + books_for_girls = 375) : 
  books_for_girls / girls * (girls + boys) = 375 := 
by
  sorry

end ms_smith_books_divided_l239_239252


namespace socks_pair_count_l239_239595

-- Define the number of socks of each color
def numWhiteSocks : Nat := 5
def numBrownSocks : Nat := 6
def numBlueSocks : Nat := 3
def numRedSocks : Nat := 2

-- Define the binomial coefficient function
noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to choose pairs of the same color
def whitePairs : Nat := binom numWhiteSocks 2
def brownPairs : Nat := binom numBrownSocks 2
def bluePairs : Nat := binom numBlueSocks 2
def redPairs : Nat := binom numRedSocks 2

-- Define the total number of ways to choose pairs of the same color
def totalPairs : Nat := whitePairs + brownPairs + bluePairs + redPairs

-- The theorem statement
theorem socks_pair_count : totalPairs = 29 := by
  sorry

end socks_pair_count_l239_239595


namespace tr_A_star_ne_neg_one_iff_I_plus_A_star_invertible_l239_239416

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (A_star : Matrix (Fin n) (Fin n) ℝ)

def adjugate (A : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ := A_star
def trace (A_star : Matrix (Fin n) (Fin n) ℝ) : ℝ := sorry -- assuming the definition of trace
def identity_matrix (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := sorry -- assuming the definition

axiom A_non_invertible : ¬ Det A ≠ 0
axiom order_ge_2 : n ≥ 2
axiom A_star_adjugate_A : A_star = adjugate A

theorem tr_A_star_ne_neg_one_iff_I_plus_A_star_invertible :
  (trace A_star ≠ -1) ↔ Invertible (identity_matrix n + A_star) :=
sorry

end tr_A_star_ne_neg_one_iff_I_plus_A_star_invertible_l239_239416


namespace hyperbola_focal_length_is_4_l239_239448

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239448


namespace find_difference_l239_239568

theorem find_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 :=
by
  sorry

end find_difference_l239_239568


namespace equidistant_from_line_CD_l239_239770

-- Definitions of points, lines, and angles
variables {Point : Type*} [metric_space Point]
variables (A B C D A' B' : Point)
variables (line : Point → Point → Set Point)
variables (angle : Point → Point → Point → Real)

-- We propose our conditions
def angles_120_degrees (A B' C D A' : Point) :=
  angle A B' C = 120 ∧ angle A B' D = 120 ∧
  angle B A' C = 120 ∧ angle B A' D = 120

def lines_intersect {P Q R S : Point} (linePQ : Set Point) (lineRS : Set Point) :=
  ∃ X, X ∈ linePQ ∧ X ∈ lineRS

-- Given conditions
variables (h1 : angles_120_degrees A B' C D A')
variables (h2 : lines_intersect (line A A') (line B B'))

-- Proof statement
theorem equidistant_from_line_CD :
  dist (A', line C D) = dist (B', line C D) :=
sorry

end equidistant_from_line_CD_l239_239770


namespace hyperbola_focal_length_proof_l239_239452

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239452


namespace dot_product_correct_l239_239567

def a : ℝ × ℝ × ℝ := (1, 1, 3)
def b : ℝ × ℝ × ℝ := (-1, 1, 2)

theorem dot_product_correct : (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 6 := by
  -- Proof goes here
  sorry

end dot_product_correct_l239_239567


namespace max_marked_vertices_no_rectangle_l239_239744

theorem max_marked_vertices_no_rectangle (n : ℕ) (hn : n = 2016) : 
  ∃ m ≤ n, m = 1009 ∧ 
  ∀ A B C D : Fin n, 
    (A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) ∧ 
    (marked A → marked B → marked C → marked D → 
     ¬is_rectangle A B C D) → 
      (∃ f : Fin n → Bool, marked f ∧ 
      (count_marked f ≤ 1009)) := sorry

end max_marked_vertices_no_rectangle_l239_239744


namespace total_balloons_cost_is_91_l239_239881

-- Define the number of balloons and their costs for Fred, Sam, and Dan
def fred_balloons : ℕ := 10
def fred_cost_per_balloon : ℝ := 1

def sam_balloons : ℕ := 46
def sam_cost_per_balloon : ℝ := 1.5

def dan_balloons : ℕ := 16
def dan_cost_per_balloon : ℝ := 0.75

-- Calculate the total cost for each person’s balloons
def fred_total_cost : ℝ := fred_balloons * fred_cost_per_balloon
def sam_total_cost : ℝ := sam_balloons * sam_cost_per_balloon
def dan_total_cost : ℝ := dan_balloons * dan_cost_per_balloon

-- Calculate the total cost of all the balloons combined
def total_cost : ℝ := fred_total_cost + sam_total_cost + dan_total_cost

-- The main statement to be proved
theorem total_balloons_cost_is_91 : total_cost = 91 :=
by
  -- Recall that the previous individual costs can be worked out and added
  -- But for the sake of this statement, we use sorry to skip details
  sorry

end total_balloons_cost_is_91_l239_239881


namespace dispatch_plans_count_l239_239668

theorem dispatch_plans_count:
  -- conditions
  let total_athletes := 9
  let basketball_players := 5
  let soccer_players := 6
  let both_players := 2
  let only_basketball := 3
  let only_soccer := 4
  -- proof
  (both_players.choose 2 + both_players * only_basketball + both_players * only_soccer + only_basketball * only_soccer) = 28 :=
by
  sorry

end dispatch_plans_count_l239_239668


namespace train_travel_distance_l239_239789

noncomputable def initial_speed : ℝ := 40  -- miles per hour
noncomputable def travel_time : ℝ := 2  -- hours
noncomputable def deceleration : ℝ := -20  -- miles per hour^2

theorem train_travel_distance : 
  let s := initial_speed * travel_time + (1/2) * deceleration * travel_time^2 in
  s = 40 :=
by
  sorry

end train_travel_distance_l239_239789


namespace hyperbola_focal_length_l239_239476

variable (m : ℝ) (h₁ : m > 0) (h₂ : ∀ x y : ℝ, (sqrt 3) * x + m * y = 0)

theorem hyperbola_focal_length :
  ∀ (m : ℝ) (h₁ : m > 0) (h₂ : sqrt 3 * x + m * y = 0), 
  sqrt m = sqrt 3 → b^2 = 1 → focal_length = 4 :=
by
  sorry

end hyperbola_focal_length_l239_239476


namespace sailboat_speed_max_power_correct_l239_239686

noncomputable def sailboat_speed_max_power
  (B S ρ v_0 v : ℝ)
  (hS : S = 7)
  (hv0 : v_0 = 6.3)
  (F : ℝ → ℝ := λ v, (B * S * ρ * (v_0 - v) ^ 2) / 2)
  (N : ℝ → ℝ := λ v, F v * v) : Prop :=
  N v = (N (6.3 / 3)) ∧ v = 6.3 / 3

theorem sailboat_speed_max_power_correct :
  sailboat_speed_max_power B S ρ 6.3 2.1 :=
by
  -- Following the above methodology and conditions, we can verify
  -- that the speed v = 6.3 / 3 satisfies the condition when the power
  -- N(v') reaches its maximum value.
  sorry

end sailboat_speed_max_power_correct_l239_239686


namespace domain_is_correct_l239_239237

def domain_of_function (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 1 ≠ 0) ∧ (x + 2 > 0)

theorem domain_is_correct :
  { x : ℝ | domain_of_function x } = { x : ℝ | -2 < x ∧ x ≤ 3 ∧ x ≠ -1 } :=
by
  sorry

end domain_is_correct_l239_239237


namespace hyperbola_focal_length_proof_l239_239453

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239453


namespace find_square_number_divisible_by_five_l239_239383

noncomputable def is_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

theorem find_square_number_divisible_by_five :
  ∃ x : ℕ, x ≥ 50 ∧ x ≤ 120 ∧ is_square x ∧ x % 5 = 0 ↔ x = 100 := by
sorry

end find_square_number_divisible_by_five_l239_239383


namespace solve_equation_l239_239661

noncomputable def equation_solution (x : ℝ) : Prop :=
  -x^3 = (2 * x^2 + 5 * x - 3) / (x - 2)

theorem solve_equation : ∀ x : ℝ, equation_solution x → x = 3 ∨ x ≈ -1.4656 :=
by
  intro x
  intro h
  sorry

end solve_equation_l239_239661


namespace train_speed_l239_239310

-- Definition of the problem
def train_length : ℝ := 350
def time_to_cross_man : ℝ := 4.5
def expected_speed : ℝ := 77.78

-- Theorem statement
theorem train_speed :
  train_length / time_to_cross_man = expected_speed :=
sorry

end train_speed_l239_239310


namespace hyperbola_focal_length_is_4_l239_239446

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) (asymptote_slope : ℝ) : ℝ :=
  if asymptote_slope = -m / (√3) ∧ asymptote_slope = √m then 4 else 0

theorem hyperbola_focal_length_is_4 (m : ℝ) (hm : m > 0) :
  (λ m, ∃ asymptote_slope : ℝ, asymptote_slope = -m / (√3) ∧ asymptote_slope = √m) m →
  hyperbola_focal_length m hm (-m / (√3)) = 4 := 
sorry

end hyperbola_focal_length_is_4_l239_239446


namespace FLikeShapeRotated180MatchesD_l239_239321

-- Definitions based on conditions
def FLikeShape := True -- Placeholder for the actual representation of "F-like shape"
def OptionA := False -- Actual content of options is abstracted
def OptionB := False
def OptionC := False
def OptionD := True  -- The correct answer as per the solution
def OptionE := False

-- The statement that needs to be proven
theorem FLikeShapeRotated180MatchesD :
  (rotate180 FLikeShape) = OptionD := 
  sorry

-- Function placeholder to represent 180-degree rotation
def rotate180 (shape : Prop) : Prop := shape -- Simplified placeholder

end FLikeShapeRotated180MatchesD_l239_239321


namespace property_tax_increase_l239_239996

noncomputable def new_assessed_value (tax_rate : ℚ) (original_value : ℚ) (tax_increase : ℚ) : ℚ :=
  let original_tax := tax_rate * original_value
  let new_tax := original_tax + tax_increase
  new_tax / tax_rate

theorem property_tax_increase :
  let tax_rate := (10 : ℚ) / 100
  let original_value := 20000
  let tax_increase := 800 in
  new_assessed_value tax_rate original_value tax_increase = 28000 :=
by 
  sorry

end property_tax_increase_l239_239996


namespace students_just_passed_l239_239766

theorem students_just_passed (total_students : ℕ) (first_div_percentage : ℕ) (second_div_percentage : ℕ) (first_div_students second_div_students just_passed_students : ℕ) :
  total_students = 300 →
  first_div_percentage = 25 →
  second_div_percentage = 54 →
  first_div_students = (first_div_percentage * total_students) / 100 →
  second_div_students = (second_div_percentage * total_students) / 100 →
  just_passed_students = total_students - (first_div_students + second_div_students) →
  just_passed_students = 63 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h7 : first_div_students = 75 := by rw [h1, h2, nat.mul_div_cancel' (show 25 ≤ 100 by linarith)]
  have h8 : second_div_students = 162 := by rw [h1, h3, nat.mul_div_cancel' (show 54 ≤ 100 by linarith)]
  rw [h4, h5, h6, h7, h8]
  sorry

end students_just_passed_l239_239766


namespace interest_earned_l239_239978

def principal_amount : ℝ := 3000
def annual_rate (y : ℝ) : ℝ := y / 100
def compounding_frequency : ℚ := 1
def time_period : ℚ := 2

theorem interest_earned (y : ℝ) : 
  let P := principal_amount in
  let r := annual_rate y in
  let n := compounding_frequency in
  let t := time_period in
  (P * ((1 + r)^t - 1) = 3000 * ((1 + (y / 100))^2 - 1)) := 
by
  sorry

end interest_earned_l239_239978


namespace officer_selection_correct_l239_239902

def members := ["Alice", "Bob", "Carol", "Dave", "Eve"]

def president : Type := String
def secretary : Type := String
def treasurer : Type := String

noncomputable def choose_officers_with_constraint : Nat :=
  (if "Dave" ∈ members then
    (Finset.card (Finset.filter (λ t, t ≠ "Dave") (Finset.powersetLen 3 (Finset.of_list members))))
      * 6 -- (3 positions can be permuted in 3! = 6 ways)
    +
    (Finset.card (Finset.filter (λ t, "Dave" ∈ t) (Finset.powersetLen 3 (Finset.of_list members))))
      * 4 -- (3 positions can be permuted with 2 possibilities for Dave, 2 positions to assign to the other two)
  else 0)

theorem officer_selection_correct : choose_officers_with_constraint = 48 :=
  sorry

end officer_selection_correct_l239_239902


namespace value_of_x_squared_plus_9y_squared_l239_239579

theorem value_of_x_squared_plus_9y_squared (x y : ℝ) (h1 : x - 3 * y = 3) (h2 : x * y = -9) : x^2 + 9 * y^2 = -45 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l239_239579


namespace smallest_solution_to_equation_l239_239034

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239034


namespace part1_daily_profit_part2_price_reduction_l239_239317

variables 
  (initial_units : ℕ) 
  (profit_per_unit : ℕ) 
  (extra_units_per_yuan_reduction : ℕ)
  (units_per_day : ℕ)
  (desired_daily_profit : ℕ)

noncomputable def daily_profit_when_selling_80_units : ℕ :=
  let increase_in_sales := units_per_day - initial_units in
  let price_reduction := increase_in_sales / extra_units_per_yuan_reduction in
  let new_profit_per_unit := profit_per_unit - price_reduction in
  new_profit_per_unit * units_per_day

theorem part1_daily_profit 
  (h1 : initial_units = 60)
  (h2 : profit_per_unit = 100)
  (h3 : extra_units_per_yan_reduction = 2)
  (h4 : units_per_day = 80) : 
  daily_profit_when_selling_80_units initial_units profit_per_unit extra_units_per_yuan_reduction units_per_day = 7200 := 
by sorry

noncomputable def price_reduction_for_8400_profit : ℕ :=
  let a := 100
  let b := 60
  let c := 8400
  let delta := (70 * 70) - 4 * 1 * 1200 -- The discriminant of the quadratic equation x^2 - 70x + 1200 = 0
  let sqrt_delta := nat.sqrt delta
  let x1 := (70 + sqrt_delta) / 2
  let x2 := (70 - sqrt_delta) / 2
  max x1 x2

theorem part2_price_reduction 
  (h1 : initial_units = 60)
  (h2 : profit_per_unit = 100)
  (h3 : extra_units_per_yan_reduction = 2)
  (h4 : desired_daily_profit = 8400) : 
  price_reduction_for_8400_profit initial_units profit_per_unit extra_units_per_yuan_reduction desired_daily_profit = 40 := 
by sorry

end part1_daily_profit_part2_price_reduction_l239_239317


namespace collinear_t1_find_a_l239_239417

def O : ℝ × ℝ := (0, 0)

def A : ℝ × ℝ := (0, 2)

def B : ℝ × ℝ := (4, 6)

def vec (P Q : ℝ × ℝ) := (Q.1 - P.1, Q.2 - P.2)

def OM (t1 t2 : ℝ) := (t1 * A.1 + t2 * (B.1 - A.1), t1 * A.2 + t2 * (B.2 - A.2))

def perpendicular (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2 = 0

def area (P Q R : ℝ × ℝ) := (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Proof problem part (1): 
theorem collinear_t1 (t2 : ℝ) : 
  let M := OM 1 t2 
  in collinear A B M := 
sorry

-- Proof problem part (2): 
theorem find_a (a : ℝ) : 
  let t1 := a^2 
      t2 := - (1 / 4) * a^2 
      M := OM t1 t2 
  in perpendicular (vec O M) (vec O AB) ∧ area A B M = 12 → a = 2 ∨ a = -2 := 
sorry

end collinear_t1_find_a_l239_239417


namespace common_root_equation_l239_239683

theorem common_root_equation (a b r : ℝ) (h₁ : a ≠ b)
  (h₂ : r^2 + 2019 * a * r + b = 0)
  (h₃ : r^2 + 2019 * b * r + a = 0) :
  r = 1 / 2019 :=
by
  sorry

end common_root_equation_l239_239683


namespace area_BCD_l239_239608

-- Definitions based on the problem's given conditions
def point := ℝ × ℝ

-- Triangle area calculation function
def triangle_area (A B C : point) : ℝ :=
  1 / 2 * ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A)).abs

-- Given points A, B, C, D
def A : point := (0, 0)
def C : point := (6, 0)
def D : point := (40, 0)
def B : point := (9, 18)

-- Given conditions
def area_ABC := 36
def AC := 6
def CD := 34

-- Prove that the area of triangle BCD is 204 square units
theorem area_BCD : triangle_area B C D = 204 := 
  sorry

end area_BCD_l239_239608


namespace find_triple_l239_239787

theorem find_triple 
  (a b c : ℚ)
  (h1 : x = 3 * real.cos t + 2 * real.sin t)
  (h2 : y = 5 * real.sin t) 
  : a = 1/9 ∧ b = -4/45 ∧ c = 124/1125 ↔
  (a * x^2 + b * x * y + c * y^2 = 4) :=
by
  sorry

end find_triple_l239_239787


namespace volume_formula_l239_239599

noncomputable def volume_of_truncated_quadrilateral_pyramid (a b : ℝ) : ℝ :=
  (ab * (a * a + a * b + b * b)) / (3 * (a + b))

theorem volume_formula (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (hlateral : lateral_surface_area = total_surface_area / 2) :
  volume_of_truncated_quadrilateral_pyramid a b = ab * (a * a + a * b + b * b) / (3 * (a + b)) :=
sorry

end volume_formula_l239_239599


namespace smallest_solution_to_equation_l239_239031

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l239_239031


namespace smallest_solution_l239_239039

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l239_239039


namespace probability_of_F_is_one_fourth_l239_239805

noncomputable def probability_of_F : ℚ :=
  let pD := (3 : ℚ) / 8
  let pE := (1 : ℚ) / 4
  let pG := (1 : ℚ) / 8
  let pF := 1 - (pD + pE + pG)
  pF

theorem probability_of_F_is_one_fourth :
  probability_of_F = 1 / 4 := by
  have h1 : (3 : ℚ) / 8 + (1 : ℚ) / 4 + (1 : ℚ) / 8 = 6 / 8
  {
    linarith,
  }
  have h2 : 1 - (6 / 8) = 1 / 4
  {
    norm_num,
  }
  have h3 : 1 - ((3 : ℚ) / 8 + (1 : ℚ) / 4 + (1 : ℚ) / 8) = 1 / 4
  {
    rw [←h1, ←h2]
  }
  exact h3

end probability_of_F_is_one_fourth_l239_239805


namespace hyper_prime_dates_count_2008_l239_239830

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def prime_months : list ℕ := [2, 3, 5, 7, 11]
def year_last_two_digits : ℕ := 8
def hyper_prime_dates_2008 : ℕ :=
  let prime_days_in_february := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] in
  prime_days_in_february.length

theorem hyper_prime_dates_count_2008 :
  (∀ m ∈ prime_months, year_last_two_digits % m ≠ 0) → hyper_prime_dates_2008 = 10 :=
by
  sorry

end hyper_prime_dates_count_2008_l239_239830


namespace smallest_abs_sum_l239_239181

open Matrix

noncomputable def matrix_square_eq (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![a, b], ![c, d]] * ![![a, b], ![c, d]]

theorem smallest_abs_sum (a b c d : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
    (M_eq : matrix_square_eq a b c d = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| = 8 :=
sorry

end smallest_abs_sum_l239_239181


namespace shorter_piece_length_l239_239314

theorem shorter_piece_length (x : ℝ) :
  (120 - (2 * x + 15) = x) → x = 35 := 
by
  intro h
  sorry

end shorter_piece_length_l239_239314


namespace line_AB_bisects_segment_DE_l239_239681

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  {trapezoid : A × B × C × D} (AC CD : Prop) (BD_sym : Prop) (intersect_E : Prop)
  (line_AB : Prop) (bisects_DE : Prop)

-- Given a trapezoid ABCD
def is_trapezoid (A B C D : Type) : Prop := sorry

-- Given the diagonal AC is equal to the side CD
def diagonal_eq_leg (AC CD : Prop) : Prop := sorry

-- Given line BD is symmetric with respect to AD intersects AC at point E
def symmetric_line_intersect (BD_sym AD AC E : Prop) : Prop := sorry

-- Prove that line AB bisects segment DE
theorem line_AB_bisects_segment_DE
  (h_trapezoid : is_trapezoid A B C D)
  (h_diagonal_eq_leg : diagonal_eq_leg AC CD)
  (h_symmetric_line_intersect : symmetric_line_intersect BD_sym (sorry : Prop) AC intersect_E)
  (h_line_AB : line_AB) :
  bisects_DE := sorry

end line_AB_bisects_segment_DE_l239_239681


namespace negation_universal_proposition_l239_239701

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by
  sorry

end negation_universal_proposition_l239_239701


namespace focal_length_of_hyperbola_l239_239487

noncomputable def hyperbola (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / m) - y^2 = 1) ∧ (∃ x y : ℝ, (sqrt 3 * x + m * y = 0))

theorem focal_length_of_hyperbola (m : ℝ) (hyp : hyperbola m) : 
  ∀ c : ℝ, c = 2 → focal_length c := 
by
  sorry

end focal_length_of_hyperbola_l239_239487


namespace mathematicians_are_saved_l239_239198

noncomputable def mathematicians_strategy_exists : Prop :=
  ∃ (strategy : ℕ → (ℕ → ℝ) → ℝ),
    let barrels : ℕ → ℝ := sorry,
    let partitions : ℕ → set ℕ := sorry,
    let sequences : (ℕ → ℝ) → Prop := sorry,
    ∀ n : ℕ, 
      (1 ≤ n ∧ n ≤ 63) →
      (∃ k : ℕ, 
        k = (max {ki | i ≠ n ∧ 1 ≤ i ∧ i ≤ 63}) →
        (strategy n barrels = barrels k) ∧
        (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 63 ∧ m ≠ n → sequences (λ i, if i = m then barrels i else sorry)))

theorem mathematicians_are_saved : mathematicians_strategy_exists :=
sorry

end mathematicians_are_saved_l239_239198


namespace sin_double_angle_neg_l239_239949

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l239_239949


namespace inequality_relationship_l239_239884

noncomputable def a : ℝ := Real.sin (4 / 5)
noncomputable def b : ℝ := Real.cos (4 / 5)
noncomputable def c : ℝ := Real.tan (4 / 5)

theorem inequality_relationship : c > a ∧ a > b := sorry

end inequality_relationship_l239_239884


namespace evaluate_expression_l239_239853

theorem evaluate_expression : ((Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5)) *
                             (Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5)) *
                             (Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5)) *
                             (Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5)) *
                             (Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5)) *
                             (Int.floor (-0.5) * Int.ceil (0.5)) * 2 = -1036800) :=
by
  have h₀ : ∀ n ∈ ({0, 1, 2, 3, 4, 5} : Set ℤ), Int.floor (-n - 0.5) * Int.ceil (n + 0.5) = -(n + 1)^2 :=
    sorry
  sorry

end evaluate_expression_l239_239853


namespace min_k_value_l239_239894

def distinct {α : Type*} (s : set α) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x = y → x = y

def T (a : ℝ) (n : ℕ) : set ℝ :=
  {b | ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ b = a + 2^j}

def k (S : set ℝ) (n : ℕ) : ℕ :=
  (⋃ a ∈ S, T a n).to_finset.card

theorem min_k_value (n : ℕ) (hn : n ≥ 2) (S : set ℝ) (h_dinstinct : distinct S ∧ S.to_finset.card = n) :
  k S n = n * (n + 1) / 2 := 
sorry

end min_k_value_l239_239894


namespace cos_of_F_in_def_l239_239613

theorem cos_of_F_in_def (E F : ℝ) (h₁ : E + F = π / 2) (h₂ : Real.sin E = 3 / 5) : Real.cos F = 3 / 5 :=
sorry

end cos_of_F_in_def_l239_239613


namespace probability_of_3_l239_239261

def sample_space : set (ℕ × ℕ) := 
  { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ x + y ≤ 6 }

def favorable_outcomes : set (ℕ × ℕ) := 
  { (x, y) ∈ sample_space | x = 3 ∨ y = 3 }

theorem probability_of_3 : ∑ (x, y) in favorable_outcomes, 1 / ∑ (x, y) in sample_space, 1 = 1 / 3 := sorry

end probability_of_3_l239_239261


namespace min_value_quadratic_expression_l239_239863

theorem min_value_quadratic_expression :
  ∃ x y : ℝ, min_val (3*x^2 + 3*x*y + y^2 - 3*x + 3*y + 9) = (45 / 8) := 
sorry

end min_value_quadratic_expression_l239_239863


namespace hyperbola_focal_length_proof_l239_239451

-- Define the hyperbola C with given equation
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1

-- Define the condition that m must be greater than 0
def m_pos (m : ℝ) := m > 0

-- Define the asymptote of the hyperbola
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- The focal length calculation based on given m
def focal_length (m : ℝ) := 2 * sqrt(m + m) -- since a^2 = m and b^2 = m

-- The proof statement that we need to prove
theorem hyperbola_focal_length_proof (m : ℝ) (h1 : m_pos m) (h2 : ∀ x y, asymptote_eq x y m) :
  focal_length m = 4 := sorry

end hyperbola_focal_length_proof_l239_239451


namespace largest_lcm_18_l239_239271

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l239_239271


namespace max_saved_houses_l239_239145

theorem max_saved_houses (n c : ℕ) (h₁ : 1 ≤ c ∧ c ≤ n / 2) : 
  ∃ k, k = n^2 + c^2 - n * c - c :=
by
  sorry

end max_saved_houses_l239_239145


namespace chord_length_intercepted_l239_239680

theorem chord_length_intercepted 
  (line_eq : ∀ x y : ℝ, 3 * x - 4 * y = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2) : 
  ∃ l : ℝ, l = 2 :=
by 
  sorry

end chord_length_intercepted_l239_239680
