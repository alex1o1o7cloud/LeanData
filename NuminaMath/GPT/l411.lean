import Mathlib

namespace exists_zero_term_in_ap_l411_411042

-- Defining the arithmetic progression and given conditions
variables (a d : ℕ) (n m : ℕ) (a_seq  : ℕ → ℤ)
-- The general term of the AP
def general_term : ℕ → ℤ := λ k, a + k * d
-- Known condition from the problem
axiom term_condition : a_seq (2*n) / a_seq (2*m) = -1

-- The main statement of the problem
theorem exists_zero_term_in_ap (a d : ℕ) (n m : ℕ)
  (a_seq  : ℕ → ℤ) (term_condition: a_seq (2*n) / a_seq (2*m) = -1) :
  ∃ k : ℕ, k = n + m ∧ a_seq (k) = 0 := 
sorry

end exists_zero_term_in_ap_l411_411042


namespace common_chord_symmedian_l411_411110

-- Definitions for points A, B, C, and the circles S1, S2
variables {A B C : Type} 
variables [metric_space A] [metric_space B] [metric_space C]
variables (S1 S2 : set ℝ)

-- Conditions on the circles
def passes_through (S : set ℝ) (P Q : Type) [metric_space P] [metric_space Q] : Prop :=
∀ (x : P), x ∈ S -> x = Q

def tangent_to (S : set ℝ) (l : set ℝ) : Prop :=
∀ (P : ℝ), P ∈ S -> P ∈ l

axiom circle_S1_conditions : passes_through S1 A B ∧ tangent_to S1 (set Icc (inf A C) (sup A C))
axiom circle_S2_conditions : passes_through S2 A C ∧ tangent_to S2 (set Icc (inf A B) (sup A B))

-- Proving the common chord AP is the symmedian of triangle ABC
theorem common_chord_symmedian {A B C : Type} [metric_space A] [metric_space B] [metric_space C] 
    (S1 S2 : set ℝ) :
    (passes_through S1 A B ∧ tangent_to S1 (set Icc (inf A C) (sup A C))) ∧
    (passes_through S2 A C ∧ tangent_to S2 (set Icc (inf A B) (sup A B))) ->
    let P := λ (x : S1 ∩ S2), x in
    symmedian AP ABC :=
begin
    -- Proof goes here
    sorry
end

end common_chord_symmedian_l411_411110


namespace inequality_proof_l411_411444

theorem inequality_proof (a b c d : ℝ) (hnonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (hsum : a + b + c + d = 1) :
  abcd + bcda + cdab + dabc ≤ 1/27 + (176/27) * abcd :=
by
  sorry

end inequality_proof_l411_411444


namespace B_and_C_together_l411_411675

theorem B_and_C_together (A B C : ℕ) (h1 : A + B + C = 1000) (h2 : A + C = 700) (h3 : C = 300) :
  B + C = 600 :=
by
  sorry

end B_and_C_together_l411_411675


namespace max_integer_points_covered_by_square_l411_411689

theorem max_integer_points_covered_by_square (n : ℕ) (S : set (ℝ × ℝ)) 
    (H1 : ∃ (x y: ℝ), (x,y) ∈ S)
    (H2 : ∀ (a b : ℝ), (a,b) ∈ S → a ≥ 0 ∧ a ≤ n ∧ b ≥ 0 ∧ b ≤ n)
    (H3 : ∀(p q : ℝ), ((p, q) ∈ S) ∧ (p ∈ ℤ) ∧ (q ∈ ℤ)) : 
  finite {p : ℤ × ℤ | p ∈ S} ∧ card {p : ℤ × ℤ | p ∈ S} ≤ (n + 1)^2 := 
by
  sorry

end max_integer_points_covered_by_square_l411_411689


namespace xy_value_l411_411794

theorem xy_value (x y : ℝ) (h1 : 4^x / 2^(x + y) = 16) (h2 : 9^(x + y) / 3^(5 * y) = 81) : x * y = 32 := 
by
  sorry

end xy_value_l411_411794


namespace tank_length_is_25_l411_411671

noncomputable def cost_to_paise (cost_in_rupees : ℕ) : ℕ :=
  cost_in_rupees * 100

noncomputable def total_area_plastered (total_cost_in_paise : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  total_cost_in_paise / cost_per_sq_m

noncomputable def length_of_tank (width height cost_in_rupees rate : ℕ) : ℕ :=
  let total_cost_in_paise := cost_to_paise cost_in_rupees
  let total_area := total_area_plastered total_cost_in_paise rate
  let area_eq := total_area = (2 * (height * width) + 2 * (6 * height) + (height * width))
  let simplified_eq := total_area - 144 = 24 * height
  (total_area - 144) / 24

theorem tank_length_is_25 (width height cost_in_rupees rate : ℕ) : 
  width = 12 → height = 6 → cost_in_rupees = 186 → rate = 25 → length_of_tank width height cost_in_rupees rate = 25 :=
  by
    intros hwidth hheight hcost hrate
    unfold length_of_tank
    rw [hwidth, hheight, hcost, hrate]
    simp
    sorry

end tank_length_is_25_l411_411671


namespace average_rainfall_l411_411221

-- Define the conditions as individual definitions in Lean
def rain_first_30_minutes : ℝ := 5
def rain_next_30_minutes : ℝ := rain_first_30_minutes / 2
def rain_next_hour : ℝ := 1 / 2
def total_duration_hours : ℝ := 2

-- Sum total amount of rain
def total_rain : ℝ := rain_first_30_minutes + rain_next_30_minutes + rain_next_hour

-- Define the theorem to prove average rainfall per hour
theorem average_rainfall : total_rain / total_duration_hours = 4 := by
  sorry

end average_rainfall_l411_411221


namespace area_triangle_PEF_l411_411440

-- Definitions of the endpoints of the major axis and foci of the ellipse
variable {A B E F P : Point}
variable h1 : |AB| = 4
variable h2 : |AF| = 2 + sqrt 3
variable h3 : LiesOnEllipse P E F
variable h4 : |PE| * |PF| = 2

-- Lean definition of the proof problem
theorem area_triangle_PEF :
  ΔArea P E F = 1 :=
sorry

end area_triangle_PEF_l411_411440


namespace smallest_integer_y_n_l411_411257

noncomputable def y : ℕ → ℝ
| 1 => real.rpow 4 (1/4)
| n + 1 => real.rpow (y n) (1/4)

theorem smallest_integer_y_n : ∃ n, y n ∈ ℤ ∧ ∀ m < n, y m ∉ ℤ :=
by
  use 4
  have y4_int : y 4 = 4 := by sorry
  constructor
  · -- y 4 is an integer
    rw y4_int
    norm_cast
  · -- For all m < 4, y m is not an integer
    intro m h_m_lt_4
    interval_cases m
    · norm_num
    · norm_num
    · norm_num
    · norm_num

end smallest_integer_y_n_l411_411257


namespace course_selection_plans_l411_411668

theorem course_selection_plans (courses : Finset ℕ) (A B : ℕ) (hA : A ∈ courses) (hB : B ∈ courses) 
  (h_card : courses.card = 8) : 
  (courses.filter (λ x, x ≠ A ∧ x ≠ B)).card = 6 → 
  (∀ s : Finset ℕ, s ⊆ courses → s.card = 5 → (s.filter (λ x, x = A)).card ≤ 1 ∧ (s.filter (λ x, x = B)).card ≤ 1) →
  ∃ plans : Finset (Finset ℕ), plans.card = 36 :=
begin
  sorry
end

end course_selection_plans_l411_411668


namespace find_k_l411_411781

noncomputable def vec_a : ℝ × ℝ := (1, 2)
noncomputable def vec_b : ℝ × ℝ := (-3, 2)
noncomputable def vec_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
noncomputable def vec_a_minus_3b : ℝ × ℝ := (10, -4)

theorem find_k :
  ∃! k : ℝ, (vec_k_a_plus_b k).1 * vec_a_minus_3b.2 = (vec_k_a_plus_b k).2 * vec_a_minus_3b.1 ∧ k = -1 / 3 :=
by
  sorry

end find_k_l411_411781


namespace number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l411_411821

def number_of_seatings (n : ℕ) : ℕ := Nat.factorial n

theorem number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other :
  let k := 2      -- Kolya and Olya as a unit
  let remaining := 3 -- The remaining people
  let pairs := 4 -- Pairs of seats that Kolya and Olya can take
  let arrangements_kolya_olya := pairs * 2 -- Each pair can have Kolya and Olya in 2 arrangements
  let arrangements_remaining := number_of_seatings remaining 
  arrangements_kolya_olya * arrangements_remaining = 48 := by
{
  -- This would be the location for the proof implementation
  sorry
}

end number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l411_411821


namespace yogurt_cost_l411_411159

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l411_411159


namespace last_two_digits_sum_factorials_l411_411966

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l411_411966


namespace sum_factorials_last_two_digits_l411_411972

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l411_411972


namespace max_value_sum_of_reciprocals_l411_411755

noncomputable def maximum_sum_of_reciprocals_of_eccentricities (F₁ F₂ P : Point) (a a₁ c e₁ e₂: ℝ) 
    (h1 : ℝ) (h2 : ℝ) (h3 : a > a₁) 
    (h4 : dist F₁ F₂ = 2 * c)
    (h5 : angle F₁ P F₂ = π / 3) 
    (h6 : eccentricity (ellipse F₁ F₂ P a) = e₁) 
    (h7 : eccentricity (hyperbola F₁ F₂ P a₁) = e₂) : 
    ℝ :=
  let r1 := dist P F₁
  let r2 := dist P F₂
  let lhs := 4 * c^2
  let rhs_ellipse := 4 * a^2 - 3 * r1 * r2
  let rhs_hyperbola := 4 * a1^2 + r1 * r2
  if lhs = rhs_ellipse ∧ lhs = rhs_hyperbola then
    (1 / e₁) + (1 / e₂)
  else
    0

theorem max_value_sum_of_reciprocals (F₁ F₂ P : Point) (a a₁ c e₁ e₂: ℝ)
    (h1 : ℝ) (h2 : ℝ) (h3 : a > a₁)
    (h4 : dist F₁ F₂ = 2 * c)
    (h5 : angle F₁ P F₂ = π / 3)
    (h6 : eccentricity (ellipse F₁ F₂ P a) = e₁)
    (h7 : eccentricity (hyperbola F₁ F₂ P a₁) = e₂) :
  (1 / e₁) + (1 / e₂) ≤ 4 * sqrt 3 / 3 := 
sorry

end max_value_sum_of_reciprocals_l411_411755


namespace probability_of_inverse_proportion_l411_411345

def points : List (ℝ × ℝ) :=
  [(0.5, -4.5), (1, -4), (1.5, -3.5), (2, -3), (2.5, -2.5), (3, -2), (3.5, -1.5),
   (4, -1), (4.5, -0.5), (5, 0)]

def inverse_proportion_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0.5, -4.5), (4.5, -0.5)), ((1, -4), (4, -1)), ((1.5, -3.5), (3.5, -1.5)), ((2, -3), (3, -2))]

theorem probability_of_inverse_proportion:
  let num_pairs := List.length points * (List.length points - 1)
  let favorable_pairs := 2 * List.length inverse_proportion_pairs
  favorable_pairs / num_pairs = (4 : ℚ) / 45 := by
  sorry

end probability_of_inverse_proportion_l411_411345


namespace geometric_sequence_sum_l411_411008

-- Let {a_n} be a geometric sequence such that S_2 = 7 and S_6 = 91. Prove that S_4 = 28

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 r : ℝ) : ℝ := a1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a1 r : ℝ) (h1 : S 2 a1 r = 7) (h2 : S 6 a1 r = 91) :
  S 4 a1 r = 28 := 
by 
  sorry

end geometric_sequence_sum_l411_411008


namespace suitable_comprehensive_survey_l411_411183

theorem suitable_comprehensive_survey :
  ¬(A = "comprehensive") ∧ ¬(B = "comprehensive") ∧ (C = "comprehensive") ∧ ¬(D = "comprehensive") → 
  suitable_survey = "C" :=
by
  sorry

end suitable_comprehensive_survey_l411_411183


namespace largest_cuts_9x9_l411_411173

theorem largest_cuts_9x9 (k : ℕ) (V E F : ℕ) (hV : V = 81) (hE : E = 4 * k) (hF : F = 1 + 2 * k)
  (hEuler : V - E + F ≥ 2) : k ≤ 21 :=
by
  sorry

end largest_cuts_9x9_l411_411173


namespace dodecahedron_blue_green_edges_l411_411170

theorem dodecahedron_blue_green_edges
  (dodecahedron : Type)
  (faces : Finset (Set dodecahedron))
  (edges : Finset (Set dodecahedron))
  (coloring : dodecahedron → Fin 4)
  (adj : dodecahedron → dodecahedron → Prop)
  [decidable_eq dodecahedron]
  [fintype dodecahedron]
  (regular_dodecahedron : ∀ f ∈ faces, ∃! adj_faces ∈ faces, adj_faces ∈ {g | adj f g})
  (adj_symm : ∀ x y, adj x y → adj y x)
  (face_colors : ∀ f g ∈ faces, adj f g → coloring f ≠ coloring g)
  (colors : Fin 4)
  (blue green : Fin 4 := 1, 2) -- Assigning 1 and 2 as blue and green
  :
  ∃ edges_blue_green_count, edges.count (λ e, ∃ f g ∈ faces, e = {f, g} ∧ adj f g ∧ coloring f = blue ∧ coloring g = green) = 5 :=
sorry

end dodecahedron_blue_green_edges_l411_411170


namespace exists_negative_value_of_f_l411_411994

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y
axiom f_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2

theorem exists_negative_value_of_f : ∃ x > 0, f x < 0 := 
sorry

end exists_negative_value_of_f_l411_411994


namespace count_divisors_with_1728_divisors_l411_411371

noncomputable def num_divisors_of_1728_exp_1728_with_1728_divisors : Nat :=
  ∑ x in (finset.range 10369).filter (λ a, (∃ b, b ∈ (finset.range 5185) ∧ (a + 1) * (b + 1) = 1728)), 1

theorem count_divisors_with_1728_divisors : 
  num_divisors_of_1728_exp_1728_with_1728_divisors = 18 := 
sorry

end count_divisors_with_1728_divisors_l411_411371


namespace exists_student_with_at_least_12_friends_l411_411526

theorem exists_student_with_at_least_12_friends 
  (students : Finset ℕ) 
  (h_card : students.card = 25) 
  (h_friends : ∀ (s : Finset ℕ), s.card = 3 → ∃ (a b : ℕ), a ≠ b ∧ a ∈ s ∧ b ∈ s ∧ a ∈ friends b)
  : ∃ (s ∈ students), Finset.card (Finset.filter (λ t, t ∈ friends s) students) ≥ 12 :=
begin
  sorry
end

end exists_student_with_at_least_12_friends_l411_411526


namespace family_ages_l411_411213

noncomputable def father_age (s : ℝ) : ℝ := s + 20
noncomputable def daughter_age (s : ℝ) : ℝ := s - 5
noncomputable def youngest_son_age (s : ℝ) : ℝ := (s - 5) / 2

theorem family_ages (S F D Y : ℝ) (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2))
  (h3 : D = S - 5)
  (h4 : Y = (S - 5) / 2) :
  S = 18 ∧ F = 38 ∧ D = 13 ∧ Y = 6.5 :=
by
  have hS : S = 18,
    from calc
      let a : ℝ := 2 * (S + 2) - (S + 22)
      0 = a : by sorry
      2 * (S + 2) = S + 22 : by sorry
      S = 18 : by sorry
  have hF : F = 38,
      from calc
        38 = S + 20 : by sorry
        F = S + 20 : by sorry
  have hD : D = 13,
      from calc
        13 = S - 5 : by sorry
        D = S - 5 : by sorry
  have hY : Y = 6.5,
      from calc
        6.5 = (S - 5) / 2 : by sorry
        Y = (S - 5) / 2 : by sorry
  exact ⟨hS, hF, hD, hY⟩
  sorry

end family_ages_l411_411213


namespace contacts_per_dollar_l411_411678

theorem contacts_per_dollar :
  (let cost_per_contact_first := 50 / 25 in
   let cost_per_contact_second := 99 / 33 in
   cost_per_contact_second > cost_per_contact_first → 3 = 3) := 
begin
  sorry
end

end contacts_per_dollar_l411_411678


namespace solution_set_inequality_l411_411319

-- Define the function and its properties as given in the problem.
variable (f : ℝ → ℝ)

-- The conditions for the function f
axiom h1 : ∀ x ∈ set.Ioi 0, x^2 * deriv f x + 2 * x * f x = real.log x / x
axiom h2 : f real.exp(1) = 1 / (4 * real.exp 1)

-- The goal is to show the solution set for a specific inequality
theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < real.exp 1} = {x : ℝ | f x + 1 / x > 5 / (4 * real.exp 1)} :=
sorry

end solution_set_inequality_l411_411319


namespace piglet_growth_period_l411_411208

theorem piglet_growth_period :
  ∃ x : ℕ, 1800 - (30 * x + 480) = 960 ∧ (∀ y, 1800 - (30 * y + 480) = 960 → y = 12) :=
begin
  use 12,
  split,
  { norm_num, },
  { intros y hy,
    linarith, }
end

end piglet_growth_period_l411_411208


namespace pyramid_volume_eq_133_1_3_l411_411475

noncomputable def volume_of_pyramid (EF FG VE : ℝ) (hEF : EF = 10) (hFG : FG = 5) (hVE : VE = 8) 
  (base_area : ℝ := EF * FG) : ℝ :=
  1/3 * base_area * VE

theorem pyramid_volume_eq_133_1_3 : volume_of_pyramid 10 5 8 = 400 / 3 := by
  sorry

end pyramid_volume_eq_133_1_3_l411_411475


namespace x_cubed_gt_y_squared_l411_411924

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l411_411924


namespace ratio_of_first_to_fourth_term_l411_411258

theorem ratio_of_first_to_fourth_term (a d : ℝ) (h1 : (a + d) + (a + 3 * d) = 6 * a) (h2 : a + 2 * d = 10) :
  a / (a + 3 * d) = 1 / 4 :=
by
  sorry

end ratio_of_first_to_fourth_term_l411_411258


namespace part1_total_ways_at_least_7_points_part2_valid_arrangements_l411_411747

theorem part1_total_ways_at_least_7_points (red_balls white_balls : ℕ) (total_draws : ℕ) (valuation_red valuation_white : ℕ):
  red_balls = 4 ∧ white_balls = 6 ∧ total_draws = 5 ∧ valuation_red = 2 ∧ valuation_white = 1 →
  (∃ (ways : ℕ), ways = 186 ∧ (ways = 
    (choose red_balls 4 * choose white_balls 1) + 
    (choose red_balls 3 * choose white_balls 2) + 
    (choose red_balls 2 * choose white_balls 3))) :=
sorry

theorem part2_valid_arrangements (red_balls white_balls : ℕ) (total_draws : ℕ) (valuation_red valuation_white : ℕ):
  red_balls = 4 ∧ white_balls = 6 ∧ total_draws = 5 ∧ valuation_red = 2 ∧ valuation_white = 1 →
  (∃ (arrangements : ℕ), arrangements = 4320 ∧ 
    (arrangements = 
      (choose red_balls 3 * choose white_balls 2) * 
      (perms [1,1,1,0,0].length [1,1,1,0,0]) - 
      (perms_dont_adjacent [1,1,1,0,0]) - 
      (perms_all_adjacent [1,1,1,0,0])) :=
sorry

end part1_total_ways_at_least_7_points_part2_valid_arrangements_l411_411747


namespace max_planes_determined_by_15_points_l411_411978

theorem max_planes_determined_by_15_points (P : Finset (EuclideanSpace ℝ 3))
  (hP : P.card = 15)
  (h_no_4_coplanar : ∀ s ⊆ P, card s = 4 → ¬AffProf 3, real).affineSpancollapse {p ∈ s}.affineIndependent)
  end)
  = sm (3,
begin
  --
  have h_comb : finset.card Fiatfinset.choose P 3 = 455 := by { simp,
--  exact sorry, 
end,


end max_planes_determined_by_15_points_l411_411978


namespace intersection_P_Q_l411_411776

def P : Set ℝ := { x | 1 ≤ Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }
def Q : Set ℕ := { 1, 2, 3 }

theorem intersection_P_Q : P ∩ Q = { 2, 3 } := by
  sorry

end intersection_P_Q_l411_411776


namespace dodecagon_diagonals_l411_411648

theorem dodecagon_diagonals : 
  let n := 12 in 
  n = 12 → ((n * (n - 3)) / 2) = 54 :=
by
  intro n
  intro h
  rw h
  -- The steps would go here
  sorry

end dodecagon_diagonals_l411_411648


namespace no_pos_real_roots_probability_l411_411255

theorem no_pos_real_roots_probability :
  let b_values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let c_values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let pairs := { (b, c) | b ∈ b_values ∧ c ∈ c_values }
  let discriminant_non_negative (b c : Int) := b^2 - 4 * c ≥ 0
  let is_negative (b : Int) := b < 0
  let valid_pair (b c : Int) := discriminant_non_negative b c ∧ is_negative b
  let total_valid_pairs := (pairs.filter valid_pair).card
  let total_pairs := pairs.card
  (total_valid_pairs = 108) ∧ (total_pairs = 121) →
  total_valid_pairs * 121 = 108 * total_pairs :=
by {
  let b_values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let c_values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let pairs := { (b, c) | b ∈ b_values ∧ c ∈ c_values }
  let discriminant_non_negative (b c : Int) := b^2 - 4 * c ≥ 0
  let is_negative (b : Int) := b < 0
  let valid_pair (b c : Int) := discriminant_non_negative b c ∧ is_negative b
  let total_valid_pairs := (pairs.filter valid_pair).card
  let total_pairs := pairs.card
  exact sorry
}

end no_pos_real_roots_probability_l411_411255


namespace original_price_calc_l411_411690

theorem original_price_calc (h : 1.08 * x = 2) : x = 100 / 54 := by
  sorry

end original_price_calc_l411_411690


namespace alex_jelly_beans_l411_411233

theorem alex_jelly_beans (initial_ounces : ℕ) (eaten_ounces : ℕ)
  (divided_piles : ℕ) (remaining_ounces : ℕ) (pile_weight : ℕ) :
  initial_ounces = 36 →
  eaten_ounces = 6 →
  divided_piles = 3 →
  remaining_ounces = initial_ounces - eaten_ounces →
  pile_weight = remaining_ounces / divided_piles →
  pile_weight = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  have h6 : remaining_ounces = 30 := by linarith [h4]
  rw [h3, h6] at h5
  have h7 : pile_weight = 10 := by linarith [h5]
  exact h7

end alex_jelly_beans_l411_411233


namespace max_knights_l411_411075

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l411_411075


namespace inequality_for_large_n_l411_411096

theorem inequality_for_large_n (n : ℕ) (hn : n > 1) : 
  (1 / Real.exp 1 - 1 / (n * Real.exp 1)) < (1 - 1 / n) ^ n ∧ (1 - 1 / n) ^ n < (1 / Real.exp 1 - 1 / (2 * n * Real.exp 1)) :=
sorry

end inequality_for_large_n_l411_411096


namespace maximum_point_of_f_l411_411921

noncomputable def f : ℝ → ℝ := λ x, (1/3) * x^3 + (1/2) * x^2 - 2 * x + 3

theorem maximum_point_of_f : ∀ x, f x ≤ f (-2) := 
by sorry

end maximum_point_of_f_l411_411921


namespace probability_two_dice_showing_1_l411_411600

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411600


namespace sin_double_angle_implies_obtuse_triangle_l411_411807

theorem sin_double_angle_implies_obtuse_triangle 
  (A B C : Type) [has_angle A] (a : A) (b : B) (c : C) 
  (h₁ : has_angle A A < π) (h₂ : sin (2 * a) < 0) : has_angle A a > π/2 :=
  sorry

end sin_double_angle_implies_obtuse_triangle_l411_411807


namespace part_I_part_II_l411_411777

-- Define the universal set
def U := Set ℝ

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) := {x : ℝ | (x - a^2 - 2) / (x - a) < 0}

-- Part I
theorem part_I (a : ℝ) (h : a = 1/2) : ¬(B a ∩ A a) = ∅ := by
  sorry

-- Part II
theorem part_II (a : ℝ) : A a ⊆ B a ↔ 
  a ∈ {x : ℝ | -1/2 ≤ x ∧ x < 1/3} ∨ {x : ℝ | 1/3 < x ∧ x ≤ (3 - Real.sqrt 5) / 2} := by
  sorry

end part_I_part_II_l411_411777


namespace chloes_test_scores_l411_411681

theorem chloes_test_scores :
  ∃ (scores : List ℕ),
  scores = [93, 92, 86, 82, 79, 78] ∧
  (List.take 4 scores).sum = 339 ∧
  scores.sum / 6 = 85 ∧
  List.Nodup scores ∧
  ∀ score ∈ scores, score < 95 :=
by
  sorry

end chloes_test_scores_l411_411681


namespace number_of_years_with_square_digit_sum_21st_century_l411_411917

def digit_sum (n : ℕ) : ℕ := n.digits.sum

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def is_year_in_21st_century_with_square_digit_sum (year : ℕ) : Prop :=
  2000 ≤ year ∧ year < 2100 ∧ is_square (digit_sum year)

theorem number_of_years_with_square_digit_sum_21st_century :
  (finset.filter is_year_in_21st_century_with_square_digit_sum (finset.range 2100)).card 
  - (finset.filter is_year_in_21st_century_with_square_digit_sum (finset.range 2000)).card = 16 :=
sorry

end number_of_years_with_square_digit_sum_21st_century_l411_411917


namespace quotient_zeros_l411_411082

def is_100_digit_number (n : ℕ) : Prop := n ≥ 10^99 ∧ n < 10^100
def is_50_digit_number (n : ℕ) : Prop := n ≥ 10^49 ∧ n < 10^50

theorem quotient_zeros (X H : ℕ) (hX : is_100_digit_number X)
  (hH : is_50_digit_number H) (no_zero_digits_X : ∀ d, d ∈ digits 10 X → d ≠ 0)
  (head_of_X : H = X / 10^50) (divisible : X % H = 0) : 
  ∃ k, X / H = 10^50 + k ∧ 1 ≤ k ∧ k ≤ 9 := by
  -- Proof goes here
  sorry

end quotient_zeros_l411_411082


namespace part1_l411_411773

theorem part1 (a : ℝ) : 
  (∃ x ∈ set.Icc (-1:ℝ) 1, x^2 + 4 * x + a - 5 = 0) → 0 ≤ a ∧ a ≤ 8 :=
sorry

end part1_l411_411773


namespace max_planes_determined_by_15_points_l411_411979

theorem max_planes_determined_by_15_points (P : Finset (EuclideanSpace ℝ 3))
  (hP : P.card = 15)
  (h_no_4_coplanar : ∀ s ⊆ P, card s = 4 → ¬AffProf 3, real).affineSpancollapse {p ∈ s}.affineIndependent)
  end)
  = sm (3,
begin
  --
  have h_comb : finset.card Fiatfinset.choose P 3 = 455 := by { simp,
--  exact sorry, 
end,


end max_planes_determined_by_15_points_l411_411979


namespace total_points_l411_411630

theorem total_points (zach_points ben_points : ℝ) (h₁ : zach_points = 42.0) (h₂ : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
  by sorry

end total_points_l411_411630


namespace area_of_smaller_circle_l411_411952

theorem area_of_smaller_circle
  (PA AB : ℝ)
  (r s : ℝ)
  (tangent_at_T : true) -- placeholder; represents the tangency condition
  (common_tangents : true) -- placeholder; represents the external tangents condition
  (PA_eq_AB : PA = AB) :
  PA = 5 →
  AB = 5 →
  r = 2 * s →
  ∃ (s : ℝ) (area : ℝ), s = 5 / (2 * (Real.sqrt 2)) ∧ area = (Real.pi * s^2) ∧ area = (25 * Real.pi) / 8 := by
  intros hPA hAB h_r_s
  use 5 / (2 * (Real.sqrt 2))
  use (Real.pi * (5 / (2 * (Real.sqrt 2)))^2)
  simp [←hPA,←hAB]
  sorry

end area_of_smaller_circle_l411_411952


namespace average_cost_price_per_meter_l411_411048

noncomputable def average_cost_per_meter (total_cost total_meters : ℝ) : ℝ :=
  total_cost / total_meters

theorem average_cost_price_per_meter :
  let silk_cost := 416.25
  let silk_meters := 9.25
  let cotton_cost := 337.50
  let cotton_meters := 7.5
  let wool_cost := 378.0
  let wool_meters := 6.0
  let total_cost := silk_cost + cotton_cost + wool_cost
  let total_meters := silk_meters + cotton_meters + wool_meters
  average_cost_per_meter total_cost total_meters = 49.75 := by
  sorry

end average_cost_price_per_meter_l411_411048


namespace coloring_exists_l411_411641

variable {Point : Type}

def S (points : Finset Point) : Finset (Point × Point) :=
  points ×ˢ points

def separates (L : Point × Point) (X Y : Point) : Prop :=
  -- Define the separation concept based on the condition that a line separates X and Y if they are on opposite sides of it.
  sorry

def d (S : Finset (Point × Point)) (X Y : Point) : ℕ :=
  (S.filter (λ L, separates L X Y)).card

theorem coloring_exists (points : Finset Point) (h₀ : points.card = 2004)
  (h₁ : ∀ {A B C : Point}, A ∈ points → B ∈ points → C ∈ points → ¬ collinear A B C) :
  ∃ coloring : Point → Bool, 
    ∀ {X Y : Point}, X ∈ points → Y ∈ points → (coloring X = coloring Y ↔ Odd (d (S points) X Y)) :=
by
  sorry

end coloring_exists_l411_411641


namespace calc_radical_power_l411_411700

theorem calc_radical_power : (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 16))) ^ 12) = 4096 := sorry

end calc_radical_power_l411_411700


namespace range_of_a_l411_411387

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l411_411387


namespace bob_needs_50_planks_l411_411249

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l411_411249


namespace cole_round_trip_time_l411_411702

theorem cole_round_trip_time (speed_to_work : ℕ) (speed_to_home : ℕ) (time_to_work_min : ℕ)
  (h1 : speed_to_work = 60) (h2 : speed_to_home = 90) (h3 : time_to_work_min = 72) :
  let time_to_work : ℝ := time_to_work_min / 60
  let distance_to_work : ℝ := speed_to_work * time_to_work
  let time_to_home : ℝ := distance_to_work / speed_to_home in
  time_to_work + time_to_home = 2 :=
by
  sorry

end cole_round_trip_time_l411_411702


namespace parallelogram_condition_l411_411084

theorem parallelogram_condition (P Q R S T : Point) (a b c : ℝ) 
  (h1 : distinct [P, Q, R, S, T])
  (h2 : collinear [P, Q, R, S, T])
  (h3 : dist P Q = a)
  (h4 : dist P R = b)
  (h5 : dist P T = c) :
  b = c - a := sorry

end parallelogram_condition_l411_411084


namespace total_toys_l411_411365

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l411_411365


namespace no_zero_in_interval_3_7_l411_411121

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 10 then 3 * 2^x - 24 else -2^(x - 5) + 126

theorem no_zero_in_interval_3_7 : ¬∃ x, 3 < x ∧ x < 7 ∧ f x = 0 :=
sorry

end no_zero_in_interval_3_7_l411_411121


namespace angle_C_in_parallelogram_l411_411827

theorem angle_C_in_parallelogram (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_B - angle_A = 40)
  (h3 : angle_C = angle_A) : angle_C = 70 :=
begin
  sorry
end

end angle_C_in_parallelogram_l411_411827


namespace sum_of_ages_l411_411060

-- Definitions based on conditions
def age_relation1 (a b c : ℕ) : Prop := a = 20 + b + c
def age_relation2 (a b c : ℕ) : Prop := a^2 = 2000 + (b + c)^2

-- The statement to be proven
theorem sum_of_ages (a b c : ℕ) (h1 : age_relation1 a b c) (h2 : age_relation2 a b c) : a + b + c = 80 :=
by
  sorry

end sum_of_ages_l411_411060


namespace arithmetic_seq_term_298_eq_100_l411_411308

-- Define the arithmetic sequence
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the specific sequence given in the problem
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 3 n

-- State the theorem
theorem arithmetic_seq_term_298_eq_100 : a_n 100 = 298 :=
by
  -- Proof will be filled in
  sorry

end arithmetic_seq_term_298_eq_100_l411_411308


namespace function_equality_l411_411379

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l411_411379


namespace black_cells_disappear_disappear_by_time_n_l411_411196

section BlackCellsDisappear

variable (Γ : Set (ℤ × ℤ) → Set (ℤ × ℤ))

-- Definitions of conditions
def is_black (n : ℕ) : Prop :=
  ∃ T < ∞, ∀ (B : Set (ℤ × ℤ)), (B.card = n) → Γ^T(B) = ∅

def disappear_time_le_n (n : ℕ) : Prop :=
  ∀ (B : Set (ℤ × ℤ)), (B.card = n) → ∃ T ≤ n, Γ^T(B) = ∅

-- Statements of the problems
theorem black_cells_disappear (n : ℕ) : is_black Γ n :=
sorry

theorem disappear_by_time_n (n : ℕ) : disappear_time_le_n Γ n :=
sorry

end BlackCellsDisappear

end black_cells_disappear_disappear_by_time_n_l411_411196


namespace 数字花园_is_1985_l411_411406

def four_digit_number (num: Nat) (tanmi: Nat) (garden: Nat) : Prop :=
  num + tanmi = 2015 ∧ tanmi + 55 = garden

theorem 数字花园_is_1985 : ∃ (num tanmi garden : Nat), four_digit_number num tanmi garden ∧ num = 1985 :=
by
  let num := 1985
  let tanmi := 30
  let garden := 85
  use num, tanmi, garden
  split
  sorry
  rfl

end 数字花园_is_1985_l411_411406


namespace fraction_of_time_at_15mph_l411_411415

variable (t5 t15 : ℝ)
variable (h1 : t5 + t15 > 0) -- Total time is positive
variable (h2 : (5 * t5 + 15 * t15) / (t5 + t15) = 10) -- Average speed condition

theorem fraction_of_time_at_15mph : t5 = t15 → t15 / (t5 + t15) = 1 / 2 := by
  intros h3
  rw h3
  calc
    t15 / (t5 + t15)
      = t15 / (t15 + t15) : by rw h3
  ... = t15 / (2 * t15)  : by rw [← two_mul t15]
  ... = 1 / 2            : by field_simp [ne_of_gt h1]

end fraction_of_time_at_15mph_l411_411415


namespace radius_of_circle_l411_411404

-- Conditions
variables (O F E : Point) (A B D C : Point)
variables (circle : Circle O)
variables (OF_perp_DC : Perpendicular OF DC) (OF_perp_AB : Perpendicular OF AB)
variables (AB_eq_8 : length AB = 8) (DC_eq_6 : length DC = 6) (EF_eq_1 : length EF = 1)

-- Proof problem statement
theorem radius_of_circle (h1 : bisects AB E) (h2 : bisects DC F) 
    (h3 : E ∈ AB) (h4 : F ∈ DC) : 
    radius circle = 5 := 
sorry

end radius_of_circle_l411_411404


namespace find_ks_l411_411142

theorem find_ks (k : ℕ) (n : ℕ) (hk : k > 0) (hk_bound : k ≤ 2020) (hn : n > 0) :
  (3^(k-1)*n+1 ∣ (((∏ i in finset.range (k * n + 1), (i+1)) / (∏ i in finset.range (n + 1), (i+1))) ^ 2)) ↔
  (k = 1 ∨ k = 3 ∨ k = 9 ∨ k = 27 ∨ k = 81 ∨ k = 243 ∨ k = 729) := 
sorry

end find_ks_l411_411142


namespace num_complex_repairs_per_month_l411_411841

def tire_repair_charge : ℕ := 20
def tire_repair_cost : ℕ := 5
def tire_repairs_per_month : ℕ := 300

def complex_repair_charge : ℕ := 300
def complex_repair_cost : ℕ := 50

def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000
def total_monthly_profit : ℕ := 3000

def profit_per_tire_repair : ℕ := tire_repair_charge - tire_repair_cost
def total_profit_from_tire_repairs : ℕ := tire_repairs_per_month * profit_per_tire_repair 
def total_known_profit : ℕ := total_profit_from_tire_repairs + retail_profit

theorem num_complex_repairs_per_month : 
  let x := (total_monthly_profit + fixed_expenses - total_known_profit) / (complex_repair_charge - complex_repair_cost) in
  x = 2 :=
by
  let x := (total_monthly_profit + fixed_expenses - total_known_profit) / (complex_repair_charge - complex_repair_cost)
  show x = 2
  sorry

end num_complex_repairs_per_month_l411_411841


namespace unique_expression_values_count_l411_411030

def satisfies_constraints (a b c d e : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ 
  d ∈ {1, 2, 3, 4, 5} ∧ e ∈ {1, 2, 3, 4, 5} ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e 

def expression_value (a b c d e : ℕ) : ℕ :=
  (a * b - c) + (d * e)

theorem unique_expression_values_count : 
  ∃ U : Finset ℕ, (∀ (a b c d e : ℕ), satisfies_constraints a b c d e → expression_value a b c d e ∈ U) ∧ U.card = 12 :=
by
  sorry

end unique_expression_values_count_l411_411030


namespace last_two_digits_of_sum_l411_411955

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l411_411955


namespace maria_gave_towels_l411_411184

def maria_towels (green_white total_left : Nat) : Nat :=
  green_white - total_left

theorem maria_gave_towels :
  ∀ (green white left given : Nat),
    green = 35 →
    white = 21 →
    left = 22 →
    given = 34 →
    maria_towels (green + white) left = given :=
by
  intros green white left given
  intros hgreen hwhite hleft hgiven
  rw [hgreen, hwhite, hleft, hgiven]
  sorry

end maria_gave_towels_l411_411184


namespace even_Tn_minus_n_l411_411054

theorem even_Tn_minus_n (n : ℕ) (hn : n > 1) (Tn : ℕ) 
  (hTn : Tn = ∑ S in (finset.powerset (finset.range n.succ)).filter (λ S, S.nonempty ∧ ((S.sum id : ℤ) % S.card : ℤ) = 0), 1) :
  (Tn - n) % 2 = 0 := 
by sorry

end even_Tn_minus_n_l411_411054


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411545

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411545


namespace certain_number_is_60_l411_411944

theorem certain_number_is_60 
  (A J C : ℕ) 
  (h1 : A = 4) 
  (h2 : C = 8) 
  (h3 : A = (1 / 2) * J) :
  3 * (A + J + C) = 60 :=
by sorry

end certain_number_is_60_l411_411944


namespace michael_should_remove_eight_scoops_l411_411868

def total_flour : ℝ := 8
def required_flour : ℝ := 6
def scoop_size : ℝ := 1 / 4

theorem michael_should_remove_eight_scoops :
  (total_flour - required_flour) / scoop_size = 8 :=
by
  sorry

end michael_should_remove_eight_scoops_l411_411868


namespace exists_fib_multiple_2007_l411_411855

def fibonacci (n : ℕ) : ℕ :=
nat.rec_on n 0 (λ n' fn, (nat.rec_on n' 1 (λ n'' ffn, fn + ffn)))

theorem exists_fib_multiple_2007 : ∃ n > 0, fibonacci n % 2007 = 0 :=
sorry

end exists_fib_multiple_2007_l411_411855


namespace measure_angle_DBA_l411_411366

-- Define a regular hexagon ABCDEF inscribed in a circle
variables {A B C D E F : Point} (h_hex : regular_hexagon_inscribed_in_circle A B C D E F)

-- Question: Measure of angle DBA
theorem measure_angle_DBA : angle_measure D B A = 30 :=
  sorry

end measure_angle_DBA_l411_411366


namespace max_knights_seated_l411_411077

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l411_411077


namespace fraction_of_tea_in_final_cup2_is_5_over_8_l411_411164

-- Defining the initial conditions and the transfers
structure CupContents where
  tea : ℚ
  milk : ℚ

def initialCup1 : CupContents := { tea := 6, milk := 0 }
def initialCup2 : CupContents := { tea := 0, milk := 3 }

def transferOneThird (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let teaTransferred := (1 / 3) * cup1.tea
  ( { cup1 with tea := cup1.tea - teaTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk } )

def transferOneFourth (cup2 : CupContents) (cup1 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup2.tea + cup2.milk
  let amountTransferred := (1 / 4) * mixedTotal
  let teaTransferred := amountTransferred * (cup2.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup2.milk / mixedTotal)
  ( { tea := cup1.tea + teaTransferred, milk := cup1.milk + milkTransferred },
    { tea := cup2.tea - teaTransferred, milk := cup2.milk - milkTransferred } )

def transferOneHalf (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup1.tea + cup1.milk
  let amountTransferred := (1 / 2) * mixedTotal
  let teaTransferred := amountTransferred * (cup1.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup1.milk / mixedTotal)
  ( { tea := cup1.tea - teaTransferred, milk := cup1.milk - milkTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk + milkTransferred } )

def finalContents (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let (cup1Transferred, cup2Transferred) := transferOneThird cup1 cup2
  let (cup1Mixed, cup2Mixed) := transferOneFourth cup2Transferred cup1Transferred
  transferOneHalf cup1Mixed cup2Mixed

-- Statement to be proved
theorem fraction_of_tea_in_final_cup2_is_5_over_8 :
  ((finalContents initialCup1 initialCup2).snd.tea / ((finalContents initialCup1 initialCup2).snd.tea + (finalContents initialCup1 initialCup2).snd.milk) = 5 / 8) :=
sorry

end fraction_of_tea_in_final_cup2_is_5_over_8_l411_411164


namespace sum_of_two_digit_numbers_with_odd_digits_l411_411519

-- Define a two-digit number whose both digits are odd
def isTwoDigitBothOddDigits (n : ℕ) : Prop :=
  n / 10 % 2 = 1 ∧ n % 10 % 2 = 1 ∧ 10 ≤ n ∧ n < 100

-- Define the sum of all two-digit numbers whose digits are both odd
def sumTwoDigitsOdd : ℕ :=
  ∑ n in Finset.range 100, if isTwoDigitBothOddDigits n then n else 0

-- State the theorem that the sum of all two-digit numbers whose digits are both odd is 1375
theorem sum_of_two_digit_numbers_with_odd_digits : sumTwoDigitsOdd = 1375 :=
by
  sorry

end sum_of_two_digit_numbers_with_odd_digits_l411_411519


namespace problem_statement_l411_411036

-- Definitions used in the math problem
def number1 := 22 / 7
def number2 := 0.303003
def number3 := Real.sqrt 27
def number4 := Real.cbrt (-64)

-- Propositions to denote rationality and irrationality
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- The theorem statement
theorem problem_statement : 
  is_rational number1 ∧ 
  is_rational number2 ∧ 
  is_irrational number3 ∧ 
  is_rational number4 :=
by
  sorry

end problem_statement_l411_411036


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411537

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411537


namespace simplify_tax_suitable_for_leonid_l411_411427

structure BusinessSetup where
  sellsFlowers : Bool
  noPriorExperience : Bool
  worksIndependently : Bool

def LeonidSetup : BusinessSetup := {
  sellsFlowers := true,
  noPriorExperience := true,
  worksIndependently := true
}

def isSimplifiedTaxSystemSuitable (setup : BusinessSetup) : Prop :=
  setup.sellsFlowers = true ∧ setup.noPriorExperience = true ∧ setup.worksIndependently = true

theorem simplify_tax_suitable_for_leonid (setup : BusinessSetup) :
  isSimplifiedTaxSystemSuitable setup := by
  sorry

#eval simplify_tax_suitable_for_leonid LeonidSetup

end simplify_tax_suitable_for_leonid_l411_411427


namespace number_divided_by_four_l411_411178

variable (x : ℝ)

theorem number_divided_by_four (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by {
  sorry
}

end number_divided_by_four_l411_411178


namespace number_of_possible_values_for_m_l411_411126

noncomputable def log_2 (x : ℝ) : ℝ := log x / log 2

theorem number_of_possible_values_for_m :
  let m := λ (x : ℕ), x > 4 ∧ x < 1800
  in (set.count (set_of m)) = 1795 :=
by
  sorry

end number_of_possible_values_for_m_l411_411126


namespace cone_prism_volume_ratio_l411_411665

noncomputable def cone_volume_ratio (π : ℝ) (r h : ℝ) : ℝ :=
  let V_cone := (1/3) * π * (3/2 * r) ^ 2 * h
  let V_prism := (3 * r) * (4 * r) * (2 * h)
  V_cone / V_prism

theorem cone_prism_volume_ratio (π r h : ℝ) :
  cone_volume_ratio π r h = π / 32 :=
by
  let V_cone := (1/3) * π * (3/2 * r) ^ 2 * h
  let V_prism := (3 * r) * (4 * r) * (2 * h)
  have h1 : V_cone = (3 * π * r ^ 2 * h) / 4 := by sorry
  have h2 : V_prism = 24 * r ^ 2 * h := by sorry
  have ratio := V_cone / V_prism
  rw [h1, h2, div_mul_eq_div, mul_assoc, mul_div_cancel_left]
  simp only [div_eq_mul_inv, inv_mul_eq_inv_mul]
  norm_num
  sorry

end cone_prism_volume_ratio_l411_411665


namespace probability_two_dice_showing_1_l411_411606

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411606


namespace kelly_can_buy_ten_pounds_of_mangoes_l411_411469

theorem kelly_can_buy_ten_pounds_of_mangoes (h : 0.5 * 1.2 = 0.60) : 12 / (2 * 0.60) = 10 :=
  by
    sorry

end kelly_can_buy_ten_pounds_of_mangoes_l411_411469


namespace exists_2000_consecutive_with_1000_red_1000_blue_l411_411713

def red_or_blue (n : ℤ) : Prop := 
  ∀ (A : set ℤ), 
  (∀ (x y : ℤ), x ∈ A → y ∈ A → x + 1 = y → ∀ z, x ≤ z → z ≤ y → z ∈ A) 
  → |A.count (λ x, x = n) - A.count (λ x, ¬ (x = n))| ≤ 1000 

theorem exists_2000_consecutive_with_1000_red_1000_blue :
  (∀ (A : set ℤ), finite A 
  → (∀ (x y : ℤ), x ∈ A → y ∈ A → x + 1 = y → ∀ z, x ≤ z → z ≤ y → z ∈ A) 
  → |A.count (λ x, red_or_blue x) - A.count (λ x, ¬ (red_or_blue x))| ≤ 1000) 
  → ∃ I : ℤ, (∀ j, (j ≥ I ∧ j < I + 2000) → 
  (∃ r b : ℤ, (I ≠ I) 
  → j.count (λ x, red_or_blue x) = 1000 
  → j.count (λ x, ¬ (red_or_blue x)) = 1000)) :=
sorry

end exists_2000_consecutive_with_1000_red_1000_blue_l411_411713


namespace sam_digits_memorized_l411_411883

-- Definitions
def carlos_memorized (c : ℕ) := (c * 6 = 24)
def sam_memorized (s c : ℕ) := (s = c + 6)
def mina_memorized := 24

-- Theorem
theorem sam_digits_memorized (s c : ℕ) (h_c : carlos_memorized c) (h_s : sam_memorized s c) : s = 10 :=
by {
  sorry
}

end sam_digits_memorized_l411_411883


namespace mangoes_combined_l411_411684

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l411_411684


namespace chantel_final_bracelets_l411_411301

-- Definitions of the conditions in Lean
def initial_bracelets_7_days := 7 * 4
def after_school_giveaway := initial_bracelets_7_days - 8
def bracelets_10_days := 10 * 5
def total_after_10_days := after_school_giveaway + bracelets_10_days
def after_soccer_giveaway := total_after_10_days - 12
def crafting_club_bracelets := 4 * 6
def total_after_crafting_club := after_soccer_giveaway + crafting_club_bracelets
def weekend_trip_bracelets := 2 * 3
def total_after_weekend_trip := total_after_crafting_club + weekend_trip_bracelets
def final_total := total_after_weekend_trip - 10

-- Lean statement to prove the final total bracelets
theorem chantel_final_bracelets : final_total = 78 :=
by
  -- Note: The proof is not required, hence the sorry
  sorry

end chantel_final_bracelets_l411_411301


namespace min_b_val_set_y0_l411_411340

theorem min_b (b : ℝ) (f : ℝ → ℝ) (hx : ∃ x > 0, f x ≥ b * x^2 + x) :
  b ≤ (5 - 2 * Real.sqrt 7) / 3 := sorry

theorem val_set_y0 (b x1 x2 y0 : ℝ) (f : ℝ → ℝ) (hf : ∃ x, f x = 0 ∧ 1 < x1 ∧ x1 < x2)
  (hx : x1 ≠ x2 ∧ ∃ l1 l2, 
    (∀ x, l1 x = (-x1^2 + 2*x1 - b) * (x - x1)) ∧ 
    (∀ x, l2 x = (-x2^2 + 2*x2 - b) * (x - x2))) 
  (hy0 : y0 = -((x1*x2 - 2*b*(x1 + x2) + 4*b^2))) :
  y0 ∈ Ioo 0 (2/9 : ℝ) := sorry

end min_b_val_set_y0_l411_411340


namespace sequence_x_value_l411_411871

theorem sequence_x_value (x : ℕ) (h1 : 3 - 1 = 2) (h2 : 6 - 3 = 3) (h3 : 10 - 6 = 4) (h4 : x - 10 = 5) : x = 15 :=
by
  sorry

end sequence_x_value_l411_411871


namespace dice_probability_l411_411593

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411593


namespace smallest_four_digit_divisible_by_4_and_5_l411_411980

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l411_411980


namespace range_of_f_inequality_on_interval_l411_411452

noncomputable def f (x : ℝ) : ℝ := sqrt (1 - x) + sqrt (1 + x)

theorem range_of_f : set.range f = set.Icc (sqrt 2) 2 :=
by
  sorry

theorem inequality_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  f x ≤ 2 - (1/4) * x^2 :=
by
  sorry

end range_of_f_inequality_on_interval_l411_411452


namespace equation_solution_count_l411_411059

theorem equation_solution_count (n : ℕ) (h_pos : n > 0)
    (h_solutions : ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 28 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ s → 2 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) :
    n = 17 ∨ n = 18 :=
sorry

end equation_solution_count_l411_411059


namespace average_salary_technicians_correct_l411_411915

section
variable (average_salary_all : ℝ)
variable (total_workers : ℕ)
variable (average_salary_rest : ℝ)
variable (num_technicians : ℕ)

noncomputable def average_salary_technicians
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : ℝ :=
  12000

theorem average_salary_technicians_correct
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : average_salary_technicians average_salary_all total_workers average_salary_rest num_technicians h1 h2 h3 h4 = 12000 :=
sorry

end

end average_salary_technicians_correct_l411_411915


namespace number_of_possible_U_l411_411859

open Finset

def U_min : ℕ := (range 60).sum (λ i, i + 10)
def U_max : ℕ := (range 60).sum (λ i, i + 91)

theorem number_of_possible_U (S : Finset ℕ) (h₁ : set.range 60 ⊆ S)
(h₂ : 10 ≤ S.min' (by sorry))
(h₃ : S.max' (by sorry) ≤ 150) :
  ∃ U, U = S.sum id ∧ 2370 ≤ U ∧ U ≤ 7230 → (number_of_possible_U = 4861) := by sorry


end number_of_possible_U_l411_411859


namespace solve_inequalities_l411_411101

theorem solve_inequalities (x : ℤ) :
  (1 ≤ x ∧ x < 3) ↔ 
  ((↑x - 1) / 2 < (↑x : ℝ) / 3 ∧ 2 * (↑x : ℝ) - 5 ≤ 3 * (↑x : ℝ) - 6) :=
by
  sorry

end solve_inequalities_l411_411101


namespace team_red_cards_l411_411019

-- Define the conditions
def team_size (n : ℕ) := n = 11
def players_without_caution (n : ℕ) := n = 5
def yellow_cards_per_cautioned_player := 1
def yellow_cards_per_red_card := 2

-- Define the proof statement
theorem team_red_cards : 
  ∀ (total_players cautioned_players yellow_cards total_red_cards : ℕ),
  team_size total_players →
  players_without_caution (total_players - cautioned_players) →
  cautioned_players * yellow_cards_per_cautioned_player = yellow_cards →
  yellow_cards / yellow_cards_per_red_card = total_red_cards →
  total_red_cards = 3 :=
by
  intros total_players cautioned_players yellow_cards total_red_cards
  assume h1 h2 h3 h4
  sorry

end team_red_cards_l411_411019


namespace domain_of_f_l411_411262

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 6*x + 8)

theorem domain_of_f :
  {x : ℝ | x ≠ 2 ∧ x ≠ 4} = { x : ℝ | ∀ y ∈ (-∞, 2) ∪ (2, 4) ∪ (4, ∞), x = y } :=
by
  sorry

end domain_of_f_l411_411262


namespace BC_length_l411_411215

/-- Definition of the parabola and its focus. -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def focus : ℝ × ℝ := (1, 0)

/-- Point A from which the light ray is emitted. -/
def A : ℝ × ℝ := (5, 4)

/-- Given the conditions, calculating the distance |BC| -/
def find_distance_BC : ℝ × ℝ → ℝ × ℝ → ℝ := 
  λ B C, real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

/-- The main theorem: given the conditions, prove that |BC| = 25/4 -/
theorem BC_length : 
  ∃ B C : ℝ × ℝ, parabola B.1 B.2 ∧ parabola C.1 C.2 ∧ 
  B = (4, 4) ∧ C = (1 / 4, -1) ∧ 
  find_distance_BC B C = 25 / 4 :=
sorry

end BC_length_l411_411215


namespace probability_two_dice_show_1_l411_411581

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411581


namespace x_cubed_gt_y_squared_l411_411925

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l411_411925


namespace sum_factorials_last_two_digits_l411_411971

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l411_411971


namespace triangle_cos_B_triangle_find_b_l411_411411

theorem triangle_cos_B (A B C : ℝ) (a b c : ℝ) (h1 : sin (A + C) = 8 * (sin (B / 2))^2) : cos B = 15 / 17 :=
sorry

theorem triangle_find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A + C) = 8 * (sin (B / 2))^2)
  (h2 : a + c = 6)
  (h3 : 1 / 2 * a * c * sin B = 2) : b = 2 :=
sorry

end triangle_cos_B_triangle_find_b_l411_411411


namespace percentage_area_covered_l411_411664

def area_of_rectangle (length width : ℝ) : ℝ := length * width

theorem percentage_area_covered :
  let A_poster := area_of_rectangle 50 100 in
  let A_picture := area_of_rectangle 20 40 in
  (A_picture / A_poster * 100) = 16 :=
by
  sorry

end percentage_area_covered_l411_411664


namespace part_a_part_b_l411_411744

open Function

-- Define the initial sequence of numbers {10, 20, ..., 100}
def initial_sequence : List ℕ := List.finRange 10 |> List.map (λ i, (i + 1) * 10)

-- Define the operation of selecting any three numbers and adding 1 to each.
def operation (seq : List ℕ) (idx1 idx2 idx3 : ℕ) : List ℕ :=
  seq.modify_nth idx1 (λ n, n + 1) |
  |> seq.modify_nth idx2 (λ n, n + 1) |
  |> seq.modify_nth idx3 (λ n, n + 1)

-- Condition for being able to make all numbers equal
def can_make_all_equal (seq : List ℕ) : Prop :=
  ∃ k, seq.all (λ x, x = k)

-- Statement for part (a): prove we can make all numbers equal
theorem part_a : can_make_all_equal initial_sequence :=
sorry

-- Condition for being able to make all numbers equal to 200
def can_make_all_200 (seq : List ℕ) : Prop :=
  seq.all (λ x, x = 200)

-- Statement for part (b): prove we can't make all numbers equal to 200
theorem part_b : ¬ can_make_all_200 initial_sequence :=
sorry

end part_a_part_b_l411_411744


namespace smallest_four_digit_divisible_by_4_and_5_l411_411981

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l411_411981


namespace log_sufficient_not_necessary_l411_411330

theorem log_sufficient_not_necessary (a b: ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (log 10 a > log 10 b) ↔ (a > b) := 
sorry

end log_sufficient_not_necessary_l411_411330


namespace probability_ratio_l411_411276

-- Conditions definitions
def total_choices := Nat.choose 50 5
def p := 10 / total_choices
def q := (Nat.choose 10 2 * Nat.choose 5 2 * Nat.choose 5 3) / total_choices

-- Statement to prove
theorem probability_ratio : q / p = 450 := by
  sorry  -- proof is omitted

end probability_ratio_l411_411276


namespace log_x2y2_l411_411788

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16 / 11 :=
by
  sorry

end log_x2y2_l411_411788


namespace red_cards_needed_l411_411015

-- Define the initial conditions
def total_players : ℕ := 11
def players_without_cautions : ℕ := 5
def yellow_cards_per_player : ℕ := 1
def yellow_cards_per_red_card : ℕ := 2

-- Theorem statement for the problem
theorem red_cards_needed (total_players = 11) (players_without_cautions = 5) 
    (yellow_cards_per_player = 1) (yellow_cards_per_red_card = 2) : (total_players - players_without_cautions) * yellow_cards_per_player / yellow_cards_per_red_card = 3 := 
by
  sorry

end red_cards_needed_l411_411015


namespace number_of_divisors_of_x_l411_411912

theorem number_of_divisors_of_x :
  let n := 2 * 3 * 5 in
  let x := n^5 in
  x = (2^5 * 3^5 * 5^5) ∧ ∃ d, d = 216 ∧ 
  (∀ (m : ℕ), m ∣ x ↔ m ∈ { k | k ∣ 2 ∧ k ∣ 3 ∧ k ∣ 5 ∨ 
                              k ∣ 3 ∧ k ∣ 5 ∧ k ∣ 2 ∨ 
                              k ∣ 5 ∧ k ∣ 2 ∧ k ∣ 3 }).
sorry

end number_of_divisors_of_x_l411_411912


namespace students_not_making_cut_l411_411943

theorem students_not_making_cut :
  let girls := 39
  let boys := 4
  let called_back := 26
  let total := girls + boys
  total - called_back = 17 :=
by
  -- add the proof here
  sorry

end students_not_making_cut_l411_411943


namespace money_left_after_transactions_l411_411882

-- Define the coin values and quantities
def dimes := 50
def quarters := 24
def nickels := 40
def pennies := 75

-- Define the item costs
def candy_bar_cost := 6 * 10 + 4 * 5 + 5
def lollipop_cost := 25 + 2 * 10 + 10 - 5 
def bag_of_chips_cost := 2 * 25 + 3 * 10 + 15
def bottle_of_soda_cost := 25 + 6 * 10 + 5 * 5 + 20 - 5

-- Define the number of items bought
def num_candy_bars := 6
def num_lollipops := 3
def num_bags_of_chips := 4
def num_bottles_of_soda := 2

-- Define the initial total money
def total_money := (dimes * 10) + (quarters * 25) + (nickels * 5) + (pennies)

-- Calculate the total cost of items
def total_cost := num_candy_bars * candy_bar_cost + num_lollipops * lollipop_cost + num_bags_of_chips * bag_of_chips_cost + num_bottles_of_soda * bottle_of_soda_cost

-- Calculate the money left after transactions
def money_left := total_money - total_cost

-- Theorem statement to prove
theorem money_left_after_transactions : money_left = 85 := by
  sorry

end money_left_after_transactions_l411_411882


namespace exist_right_triangle_l411_411317

open Set

structure Circle (center : Point) (radius : ℝ) :=
(mem : ∀ P : Point, dist center P = radius)

variables {P : Type} [MetricSpace P] [EuclideanSpace P]

def isRightTriangleInscribed (C : Circle) (A B : Point) : Prop :=
  ∃ C : Point, C ∈ C ∧
               angle A C B = 90

theorem exist_right_triangle (C : Circle) (A B : Point) (hA : A ∈ C) (hB : B ∈ C) :
  isRightTriangleInscribed C A B :=
sorry

end exist_right_triangle_l411_411317


namespace Tom_runs_60_miles_in_a_week_l411_411535

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l411_411535


namespace every_integer_as_sum_of_squares_l411_411095

theorem every_integer_as_sum_of_squares (n : ℤ) : ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ n = (x^2 : ℤ) + (y^2 : ℤ) - (z^2 : ℤ) :=
by sorry

end every_integer_as_sum_of_squares_l411_411095


namespace trigonometric_identity_l411_411760

variables {α : ℝ}

theorem trigonometric_identity (hα : (tan α = 3 / 5) ∧ (α ∈ set.Icc (3 * π / 2) (2 * π))) :
  (1 + sqrt 2 * tan (2 * α - π / 4)) / (tan (α + π / 2)) = -2 / 5 :=
sorry

end trigonometric_identity_l411_411760


namespace last_two_digits_of_sum_of_first_50_factorials_l411_411960

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l411_411960


namespace abes_total_budget_l411_411230

theorem abes_total_budget
    (B : ℝ)
    (h1 : B = (1/3) * B + (1/4) * B + 1250) :
    B = 3000 :=
sorry

end abes_total_budget_l411_411230


namespace suburban_cities_bound_l411_411646

theorem suburban_cities_bound (n : ℕ) (h : n ≥ 2)
  (N : Fin n → Fin n)
  (closer : ∀ i j : Fin n, i ≠ j → dist (N i) i < dist j i)
  (connected : ∀ a b : Fin n, ∃ path : List (Fin n), path ≠ [] ∧ path.head = a ∧ path.reverse.head = b ∧ 
    ∀ k ∈ path, k > 0 → (k-1, k) ∈ (λ i, (i, N i)) '' set.univ) :
  ∃ k, 4 * k + 2 ≤ n ∧ {x : Fin n | ∃ i : Fin n, N i = x}.card ≥ k :=
sorry

end suburban_cities_bound_l411_411646


namespace probability_two_ones_in_twelve_dice_l411_411548

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411548


namespace shortest_path_length_l411_411408

noncomputable def A := (0, 0)
noncomputable def E := (20, 21)
noncomputable def P := (10, 10.5)
noncomputable def r := 6

def circle_eq (x y : ℝ) : Prop := (x - P.1) ^ 2 + (y - P.2) ^ 2 = r ^ 2

theorem shortest_path_length :
  ∀ (path : ℝ × ℝ → Prop), 
  (∀ (x y : ℝ), path (x, y) → ¬circle_eq x y) → 
  (path A) → 
  (path E) → 
  length_of_path path = 26.4 + 1.5 * real.pi 
  :=
sorry

end shortest_path_length_l411_411408


namespace max_value_of_x_plus_one_over_x_l411_411143

theorem max_value_of_x_plus_one_over_x :
  (∃ (n : ℕ) (a : Fin n → ℝ), n = 1009 ∧ (∀ i, 0 < a i) ∧ ∑ i, a i = 1010 ∧ ∑ i, (a i)⁻¹ = 1010) →
  ∃ x, x ∈ (λ i : Fin 1009, (a i) + (a i)⁻¹).range ∧ x = 2029 / 1010 :=
by
  sorry

end max_value_of_x_plus_one_over_x_l411_411143


namespace sum_div_ineq_l411_411745

theorem sum_div_ineq (n : ℕ) (x : ℕ → ℝ) (hx_sum_abs : ∑ i in finset.range n, |x i| = 1) 
(hx_sum_zero : ∑ i in finset.range n, x i = 0) (hn : 2 ≤ n) : 
|∑ i in finset.range n, x i / (i + 1)| ≤ 1 / 2 - 1 / (2 * n) := 
sorry

end sum_div_ineq_l411_411745


namespace probability_two_dice_showing_1_l411_411605

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411605


namespace mark_card_sum_l411_411455

/--
Mark has seven green cards numbered 1 through 7 and five red cards numbered 2 through 6.
He arranges the cards such that colors alternate and the sum of each pair of neighboring cards forms a prime.
Prove that the sum of the numbers on the last three cards in his stack is 16.
-/
theorem mark_card_sum {green_cards : Fin 7 → ℕ} {red_cards : Fin 5 → ℕ}
  (h_green_numbered : ∀ i, 1 ≤ green_cards i ∧ green_cards i ≤ 7)
  (h_red_numbered : ∀ i, 2 ≤ red_cards i ∧ red_cards i ≤ 6)
  (h_alternate : ∀ i, i < 6 → (∃ j k, green_cards j + red_cards k = prime) ∨ (red_cards j + green_cards k = prime)) :
  ∃ s, s = 16 := sorry

end mark_card_sum_l411_411455


namespace school_accomodation_proof_l411_411667

theorem school_accomodation_proof
  (total_classrooms : ℕ) 
  (fraction_classrooms_45 : ℕ) 
  (fraction_classrooms_38 : ℕ)
  (fraction_classrooms_32 : ℕ)
  (fraction_classrooms_25 : ℕ)
  (desks_45 : ℕ)
  (desks_38 : ℕ)
  (desks_32 : ℕ)
  (desks_25 : ℕ)
  (student_capacity_limit : ℕ) :
  total_classrooms = 50 ->
  fraction_classrooms_45 = (3 / 10) * total_classrooms -> 
  fraction_classrooms_38 = (1 / 4) * total_classrooms -> 
  fraction_classrooms_32 = (1 / 5) * total_classrooms -> 
  fraction_classrooms_25 = (total_classrooms - fraction_classrooms_45 - fraction_classrooms_38 - fraction_classrooms_32) ->
  desks_45 = 15 * 45 -> 
  desks_38 = 12 * 38 -> 
  desks_32 = 10 * 32 -> 
  desks_25 = fraction_classrooms_25 * 25 -> 
  student_capacity_limit = 1800 -> 
  fraction_classrooms_45 * 45 +
  fraction_classrooms_38 * 38 +
  fraction_classrooms_32 * 32 + 
  fraction_classrooms_25 * 25 = 1776 + sorry
  :=
sorry

end school_accomodation_proof_l411_411667


namespace bread_slices_leftover_l411_411093

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end bread_slices_leftover_l411_411093


namespace angle_B_value_l411_411834

theorem angle_B_value (a b c : ℝ) (h : cos (b) / cos (c) = -b / (2 * a + c)) :
  b = 2 * Math.pi / 3 ∨ b = 120 :=
sorry

end angle_B_value_l411_411834


namespace area_fyg_is_correct_l411_411166

-- Given conditions from the problem
variables (EF GH area_of_trapezoid angle_EFGH_to_parallel angle_30_deg : ℝ)
variables (is_trapezoid : EF = 15 ∧ GH = 25 ∧ area_of_trapezoid = 400
                         ∧ angle_EFGH_to_parallel = 30)

-- Define the required angle in radians for computation
noncomputable def degree_to_radian (deg : ℝ) : ℝ := deg * (Real.pi / 180)

-- Define the area of triangle FYG using the given conditions
example : ℝ :=
  let h := (2 * area_of_trapezoid) / (EF + GH) in -- height of the trapezoid
  let h' := h * (Real.cos (degree_to_radian angle_30_deg)) in -- actual height considering the angle
  let area_FYG := (3/5) * (area_of_trapezoid - (1/2) * EF * h') in -- area of triangle FYG
  area_FYG

-- State the theorem to prove the area of triangle FYG
theorem area_fyg_is_correct : 
  is_trapezoid → 
  angle_EFGH_to_parallel = 30 →
  EF = 15 →
  GH = 25 →
  area_of_trapezoid = 400 →
  Real.cos (degree_to_radian 30) = (Real.sqrt 3 / 2) →

  let h := (2 * area_of_trapezoid) / (EF + GH) in
  let h' := h * (Real.cos (degree_to_radian angle_30)) in
  let area_FYG := (3 / 5) * (area_of_trapezoid - (1 / 2) * EF * h') in

  area_FYG = 240 - 45 * Real.sqrt 3 :=
sorry

end area_fyg_is_correct_l411_411166


namespace prob_two_ones_in_twelve_dice_l411_411570

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411570


namespace shift_right_by_pi_six_l411_411532

def f (x : ℝ) : ℝ := Real.sin (2 * x)
def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem shift_right_by_pi_six :
  ∀ x, g(x) = f(x - Real.pi / 6) :=
by
  intro x
  sorry

end shift_right_by_pi_six_l411_411532


namespace crocodile_can_move_anywhere_iff_even_l411_411201

def is_even (n : ℕ) : Prop := n % 2 = 0

def can_move_to_any_square (N : ℕ) : Prop :=
∀ (x1 y1 x2 y2 : ℤ), ∃ (k : ℕ), 
(x1 + k * (N + 1) = x2 ∨ y1 + k * (N + 1) = y2)

theorem crocodile_can_move_anywhere_iff_even (N : ℕ) : can_move_to_any_square N ↔ is_even N :=
sorry

end crocodile_can_move_anywhere_iff_even_l411_411201


namespace count_distinct_four_digit_numbers_from_2025_l411_411367

-- Define the digits 2, 0, 2, and 5 as a multiset
def digits : Multiset ℕ := {2, 0, 2, 5}

-- Define the set of valid four-digit numbers formed from the digits
def four_digit_numbers (d : Multiset ℕ) : Finset ℕ :=
  Finset.filter (λ x : ℕ, 1000 ≤ x ∧ x < 10000) (Multiset.permutations d).to_finset.map
    (λ ds, ds.foldr (λ a b, a + 10 * b) 0)

-- The theorem we aim to prove
theorem count_distinct_four_digit_numbers_from_2025 : 
  (four_digit_numbers digits).card = 7 :=
sorry

end count_distinct_four_digit_numbers_from_2025_l411_411367


namespace polygon_area_l411_411828

theorem polygon_area (n : ℕ) (s l : ℕ) (h_sides : n = 24) (h_perpendicular : true) (h_congruent : true) (h_perimeter : l = 48) (h_side_length : s = 48 / 24) : 
  let area := 32 * s^2 
  in area = 128 :=
by
  -- All necessary definitions and assumptions have been included
  let s := 48 / 24 -- side length
  let area := 32 * s^2 -- calculation of the area using the structure of smaller squares
  show area = 128, by 
  -- The actual calculation confirming the area
  sorry

end polygon_area_l411_411828


namespace geometric_sequence_inequality_l411_411826

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)

-- Conditions
def geometric_sequence (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ) : Prop :=
  a₂ = a₁ * q ∧
  a₃ = a₁ * q^2 ∧
  a₄ = a₁ * q^3 ∧
  a₅ = a₁ * q^4 ∧
  a₆ = a₁ * q^5 ∧
  a₇ = a₁ * q^6 ∧
  a₈ = a₁ * q^7

theorem geometric_sequence_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)
  (h_seq : geometric_sequence a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q)
  (h_a₁_pos : 0 < a₁)
  (h_q_ne_1 : q ≠ 1) :
  a₁ + a₈ > a₄ + a₅ :=
by 
-- Proof omitted
sorry

end geometric_sequence_inequality_l411_411826


namespace length_of_X1_bisector_l411_411823

-- Definitions for right triangles and angle bisectors
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def angle_bisector (D D1 E F : ℝ) : Prop := D1 = (3 / 4) * (E + F)

-- Given right triangles
def DEF := right_triangle 5 12 13
def EF := 12  -- from 5-12-13 triangle

-- Definitions specific to the triangles in the problem
def D1E := (3 / 4) * EF
def D1F := EF - D1E

def XYZ := right_triangle D1F D1E YZ
noncomputable def YZ : ℝ := real.sqrt (D1E^2 - D1F^2)

def X1_length (XZ XY YZ : ℝ) : ℝ := (XZ / XY) * YZ

-- The theorem to state the problem
theorem length_of_X1_bisector (D E F D1 XY XZ YZ : ℝ) 
  (H1 : DEF) (H2 : angle_bisector D D1 E F) (XY_eq : XY = D1E) 
  (XZ_eq : XZ = D1F) (H3 : XYZ) :
  X1_length XZ XY YZ = (3 * real.sqrt 2) / 2 :=
by
  sorry

end length_of_X1_bisector_l411_411823


namespace solve_for_x_l411_411098

theorem solve_for_x (x : ℝ) (h : 3 * (5 ^ x) = 1875) : x = 4 :=
by sorry

end solve_for_x_l411_411098


namespace simplify_and_rationalize_l411_411900

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411900


namespace mass_percentage_H3BO3_l411_411288

theorem mass_percentage_H3BO3 :
  ∃ (element : String) (mass_percent : ℝ), 
    element ∈ ["H", "B", "O"] ∧ 
    mass_percent = 4.84 ∧ 
    mass_percent = 4.84 :=
sorry

end mass_percentage_H3BO3_l411_411288


namespace football_game_monday_fewer_people_monday_l411_411148

-- Let's define the conditions and statement
axiom saturday : ℕ := 80
axiom expected_total_audience : ℕ := 350
axiom additional_attendance : ℕ := 40

def total_audience : ℕ := expected_total_audience + additional_attendance
def wednesday (monday : ℕ) : ℕ := monday + 50
def friday (monday : ℕ) : ℕ := saturday + monday

theorem football_game_monday (monday : ℕ) :
  80 + monday + (monday + 50) + (80 + monday) = 390 →
  monday = 60 :=
by
  sorry

theorem fewer_people_monday (monday : ℕ) :
  monday = 60 →
  saturday - monday = 20 :=
by
  sorry

end football_game_monday_fewer_people_monday_l411_411148


namespace factor_expression_l411_411736

theorem factor_expression (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2) ^ 2 :=
by sorry

end factor_expression_l411_411736


namespace midpoint_property_l411_411327

structure IsoscelesTriangle (α : Type*) [EuclideanGeometry α] :=
(A B C : α)
(AB_eq_BC : dist A B = dist B C)

theorem midpoint_property
  {α : Type*} [EuclideanGeometry α]
  (T : IsoscelesTriangle α)
  (C' : α)
  (CC'_is_diameter : is_diameter_of_circumcircle T.A T.B T.C C')
  (M P : α)
  (M_on_AB : collinear T.A T.B M)
  (P_on_AC : collinear T.A T.C P)
  (C'_line_parallel_BC : parallel (line C' P) (line T.B T.C))
  (M_on_parallel_line : collinear M (line C' P))
: midpoint M C' P :=
sorry

end midpoint_property_l411_411327


namespace option_c_holds_l411_411026

noncomputable def k (n : ℝ) : ℝ := 2 * (n - 1)
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem option_c_holds (h : ∀ (n : ℝ), n ≠ 1) : 
  ∃ (n : ℝ), ∀ (x : ℝ), x > 0 → (inverse_proportion (k n - 1) x < 0) → 
  (∀ x1 x2 : ℝ, x1 < x2 → inverse_proportion (k n) x1 < inverse_proportion (k n) x2) :=
begin
  sorry -- The proof is omitted as per the instruction.
end

end option_c_holds_l411_411026


namespace irrational_sqrt_27_l411_411033

noncomputable def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

noncomputable def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem irrational_sqrt_27 :
  let a := (22 : ℚ) / (7 : ℚ)
  let b := 0.303003
  let c := real.sqrt 27
  let d := - real.cbrt 64
  (is_irrational c ∧ is_rational a ∧ is_rational b ∧ is_rational d) :=
by
  let a := (22 : ℚ) / (7 : ℚ)
  let b := 0.303003
  let c := real.sqrt 27
  let d := - real.cbrt 64
  have ha : is_rational a := sorry
  have hb : is_rational b := sorry
  have hc : is_irrational c := sorry
  have hd : is_rational d := sorry
  exact ⟨hc, ha, hb, hd⟩

end irrational_sqrt_27_l411_411033


namespace ways_to_select_5_balls_l411_411735

theorem ways_to_select_5_balls (balls : Finset ℕ) (h1 : balls = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) :
  (∑ n in balls.choose 5, if n.sum % 2 = 1 then 1 else 0) = 236 :=
by sorry

end ways_to_select_5_balls_l411_411735


namespace triangle_perimeter_l411_411001

-- Let the lengths of the sides of the triangle be a, b, c.
variables (a b c : ℕ)
-- To represent the sides with specific lengths as stated in the problem.
def side1 := 2
def side2 := 5

-- The condition that the third side must be an odd integer.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Setting up the third side based on the given conditions.
def third_side_odd (c : ℕ) : Prop := 3 < c ∧ c < 7 ∧ is_odd c

-- The perimeter of the triangle.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove.
theorem triangle_perimeter (c : ℕ) (h_odd : third_side_odd c) : perimeter side1 side2 c = 12 :=
by
  sorry

end triangle_perimeter_l411_411001


namespace triangle_b_c_sum_range_l411_411832

theorem triangle_b_c_sum_range 
  (b c : ℝ) 
  (a : ℝ := √3 / 2) 
  (h1 : b^2 + c^2 - a^2 = b * c) 
  (h2 : vector.dot_product ↑⟨1, 0⟩ ↑⟨1, 1⟩ > 0) : 
  b + c ∈ set.Ioo (3 / 2 : ℝ) (3 * √3 / 2 : ℝ) :=
sorry

end triangle_b_c_sum_range_l411_411832


namespace sum_of_squares_of_consecutive_even_integers_l411_411928

theorem sum_of_squares_of_consecutive_even_integers (n : ℤ) (h : (2 * n - 2) * (2 * n) * (2 * n + 2) = 12 * ((2 * n - 2) + (2 * n) + (2 * n + 2))) :
  (2 * n - 2) ^ 2 + (2 * n) ^ 2 + (2 * n + 2) ^ 2 = 440 :=
by
  sorry

end sum_of_squares_of_consecutive_even_integers_l411_411928


namespace probability_two_dice_show_1_l411_411578

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411578


namespace count_valid_bases_l411_411303

def is_valid_base (b : ℕ) : Prop :=
  2 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0

theorem count_valid_bases :
  (Finset.range 9).filter (λ b, is_valid_base (b + 2)).card = 6 :=
by
  sorry

end count_valid_bases_l411_411303


namespace scientific_notation_of_20000_l411_411106

def number : ℕ := 20000

theorem scientific_notation_of_20000 : number = 2 * 10 ^ 4 :=
by
  sorry

end scientific_notation_of_20000_l411_411106


namespace soccer_red_cards_l411_411020

theorem soccer_red_cards (total_players yellow_carded_players : ℕ)
  (no_caution_players : total_players = 11)
  (five_no_cautions : 5 = 5)
  (players_received_yellow : yellow_carded_players = total_players - 5)
  (yellow_per_player : ∀ p, p = 6 -> 6 * 1)
  (red_card_rule : ∀ y, (yellow_carded_players * 1) = y -> y / 2 = 3) :
  ∃ red_cards : ℕ, red_cards = 3 :=
by { 
  existsi 3,
  sorry
}

end soccer_red_cards_l411_411020


namespace perpendicular_lines_l411_411266

theorem perpendicular_lines (a : ℝ) : 
  (3 * y + x + 4 = 0) → 
  (4 * y + a * x + 5 = 0) → 
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → - (1 / 3 : ℝ) * - (a / 4 : ℝ) = -1) → 
  a = -12 := 
by
  intros h1 h2 h_perpendicularity
  sorry

end perpendicular_lines_l411_411266


namespace angle_KDA_l411_411930

theorem angle_KDA (A B C D M K : Type) [has_coe ℝ ℂ]
  (AD_eq_2AB : A = 2 * B)
  (M_mid_AD : M = (A + D) / 2)
  (angle_AMK_80 : angle A M K = 80)
  (bisects_MKC : ray K D bisects angle M K C)
  : angle K D A = 35 :=
sorry

end angle_KDA_l411_411930


namespace conjugate_of_complex_l411_411112

noncomputable def conjugate_problem (z : ℂ) (a b : ℂ) (w : ℂ) : Prop :=
  z = a / b → a = 3 - complex.i ∧ b = 1 - 2 * complex.i ∧ w = complex.conj z ∧ w = 1 - complex.i

theorem conjugate_of_complex :
  ∀ (z : ℂ), ∃ w : ℂ, conjugate_problem z (3-complex.i) (1-2*complex.i) w :=
begin
  intros z,
  use complex.conj z,
  split,
  { -- Here, z = (3 - i) / (1 - 2i)
    rintro ⟨h⟩,
    sorry },
  { -- Here, z is (3 - i) / (1 - 2i)
    sorry },
  { -- Here, w = complex.conj z
    sorry },
  { -- Finally, w should be 1 - i
    sorry }
end

end conjugate_of_complex_l411_411112


namespace product_lcm_gcd_eq_108_l411_411290

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem product_lcm_gcd_eq_108 (a b : ℕ) (h1 : a = 12) (h2 : b = 9) :
  (lcm a b) * (Nat.gcd a b) = 108 := by
  rw [h1, h2] -- replace a and b with 12 and 9
  have lcm_12_9 : lcm 12 9 = 36 := sorry -- find the LCM of 12 and 9
  have gcd_12_9 : Nat.gcd 12 9 = 3 := sorry -- find the GCD of 12 and 9
  rw [lcm_12_9, gcd_12_9]
  norm_num -- simplifies the multiplication
  exact eq.refl 108

end product_lcm_gcd_eq_108_l411_411290


namespace expected_total_score_l411_411012

theorem expected_total_score (n : ℕ) (p : ℝ) (students : ℕ) (score_2 : ℕ) (score_1 : ℕ) (score_0 : ℕ) :
  n = 2 → 
  p = 0.6 → 
  students = 10 → 
  score_2 = 10 →
  score_1 = 5 →
  score_0 = 0 →
  E(X) = 60 :=
by
  sorry

end expected_total_score_l411_411012


namespace fraction_addition_l411_411936

/--
The value of 2/5 + 1/3 is 11/15.
-/
theorem fraction_addition :
  (2 / 5 : ℚ) + (1 / 3) = 11 / 15 := 
sorry

end fraction_addition_l411_411936


namespace simplify_and_rationalize_l411_411889

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411889


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411541

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411541


namespace arithmetic_geometric_mean_l411_411486

theorem arithmetic_geometric_mean (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a + b) / 2 = m * Real.sqrt (a * b)) :
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) :=
by
  sorry

end arithmetic_geometric_mean_l411_411486


namespace james_eats_slices_l411_411840

theorem james_eats_slices :
  let initial_slices := 8 in
  let friend_eats := 2 in
  let remaining_slices := initial_slices - friend_eats in
  let james_eats := remaining_slices / 2 in
  james_eats = 3 := 
by 
  sorry

end james_eats_slices_l411_411840


namespace vector_dot_product_sum_l411_411355

variables (A B C : ℝ^3) -- Assuming the points are in 3-dimensional space

def vect_len (v : ℝ^3) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def AB := B - A
noncomputable def BC := C - B
noncomputable def CA := A - C

noncomputable def dot_product (v w : ℝ^3) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Conditions
axiom ab_len : vect_len AB = 3
axiom bc_len : vect_len BC = 4
axiom ca_len : vect_len CA = 5

-- The statement to prove
theorem vector_dot_product_sum :
  dot_product AB BC + dot_product BC CA + dot_product CA AB = -25 := 
sorry

end vector_dot_product_sum_l411_411355


namespace roots_cubic_reciprocal_l411_411438

noncomputable theory
open Complex Polynomial

-- Define the quadratic polynomial and its roots
def quadratic_poly : Polynomial ℂ := Polynomial.C 3 * X ^ 2 + Polynomial.C 5 * X + Polynomial.C 2

def roots (p : Polynomial ℂ) := if h : p.degree = 2 then p.roots else ∅

-- Conditions: r and s are the roots of the polynomial
def r := roots quadratic_poly[0]
def s := roots quadratic_poly[1]

-- The proof statement
theorem roots_cubic_reciprocal :
  (∃ r s : ℂ, is_root quadratic_poly r ∧ is_root quadratic_poly s) →
  (1 / r ^ 3 + 1 / s ^ 3 = (25 : ℝ) / 8) :=
by
  intros hr hs
  sorry

end roots_cubic_reciprocal_l411_411438


namespace milk_level_lowered_l411_411504

noncomputable def lower_milk_level (length_ft : ℝ) (width_ft : ℝ) (milk_gallons : ℝ) : ℝ :=
  let gallon_to_cubic_inches := 231
  let foot_to_inches := 12
  let box_length_inches := length_ft * foot_to_inches
  let box_width_inches := width_ft * foot_to_inches
  let box_base_area := box_length_inches * box_width_inches
  let milk_volume_cubic_inches := milk_gallons * gallon_to_cubic_inches
  milk_volume_cubic_inches / box_base_area

theorem milk_level_lowered :
  lower_milk_level 62 25 5812.5 ≈ 6.018 :=
by
  -- Proof omitted
  sorry

end milk_level_lowered_l411_411504


namespace not_certain_event_l411_411990

theorem not_certain_event :
  ∀ (A B C D : Prop), 
  (A ↔ ∀ Δ : triangle, shortest_perpendicular_segment(Δ)) →
  (B ↔ ∀ (l1 l2 : line) (t : transversal l1 l2), corresponding_angles_equal(l1, l2, t) → parallel_lines(l1, l2)) → 
  (C ↔ ∀ Δ : triangle, is_isosceles(Δ) → base_angles_equal(Δ)) → 
  (D ↔ ∀ Δ : triangle, triangle_inequality_holds(Δ)) → 
  ¬(B).

-- Definitions required for the statement (axiomatically stated for context):

axiom is_isosceles (Δ : Type) : Prop
axiom base_angles_equal (Δ : Type) : Prop
axiom triangle (Δ : Type) : Type
axiom shortest_perpendicular_segment (Δ : Type) : Prop
axiom line (l : Type) : Type
axiom transversal (l1 l2 : Type) : Type
axiom corresponding_angles_equal (l1 l2 t : Type) : Prop
axiom parallel_lines (l1 l2 : Type) : Prop
axiom triangle_inequality_holds (Δ : Type) : Prop

-- Proof
sorry

end not_certain_event_l411_411990


namespace shelves_needed_l411_411461

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l411_411461


namespace percentage_very_satisfactory_l411_411225

-- Definitions based on conditions
def total_parents : ℕ := 120
def needs_improvement_count : ℕ := 6
def excellent_percentage : ℕ := 15
def satisfactory_remaining_percentage : ℕ := 80

-- Theorem statement
theorem percentage_very_satisfactory 
  (total_parents : ℕ) 
  (needs_improvement_count : ℕ) 
  (excellent_percentage : ℕ) 
  (satisfactory_remaining_percentage : ℕ) 
  (result : ℕ) : result = 16 :=
by
  sorry

end percentage_very_satisfactory_l411_411225


namespace max_n_divisibility_l411_411975

/-
Given n is a three-digit positive integer, prove the maximum n for which
n * (2 * n + 1) does not divide n!
-/
theorem max_n_divisibility :
  ∃ n ∈ {k | 100 ≤ k ∧ k ≤ 999}, ∀ m ∈ {j | 100 ≤ j ∧ j ≤ 999}, n ≤ m → ¬ (m! % (m * (2 * m + 1)) = 0) := sorry

end max_n_divisibility_l411_411975


namespace max_grid_size_l411_411652

theorem max_grid_size (n : ℕ) : 
  (∀ (grid : Fin n → Fin n → bool), 
    ∀ (i1 i2 j1 j2 : Fin n),
      i1 < i2 → j1 < j2 →
      (grid i1 j1 = grid i1 j2) ∨
      (grid i1 j1 = grid i2 j1) ∨
      (grid i2 j1 = grid i2 j2) ∨
      (grid i1 j2 = grid i2 j2)) → n ≤ 4 :=
by
  sorry

end max_grid_size_l411_411652


namespace simplify_rationalize_denominator_l411_411908

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411908


namespace triangle_length_AC_l411_411810

theorem triangle_length_AC (A B C D F : Type) 
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (h_AB : ∀ x : A, ∀ y : B, ⟪x, y⟫ = 0) -- AB ⊥ AC
  (h_AF : ∀ x : A, ∀ z : F, ⟪x, z⟫ = 0) -- AF ⊥ BC
  (BD DF FC: ℝ) (h_BD : BD = 1) (h_DF : DF = 1) (h_FC : FC = 1)
  (h_midpoint : ∀ AD DC : ℝ, AD = DC / 2) 
  (h_D_on_AC : ∀ x : D, x ∈ line_through A C) 
  (h_F_on_BC : ∀ x : F, x ∈ line_through B C) :
  ∃ x : ℝ, x = sqrt 2 := 
sorry

end triangle_length_AC_l411_411810


namespace acute_angle_measure_l411_411149

theorem acute_angle_measure (r1 r2 r3 : ℝ) (θ : ℝ)  :
  (r1 = 4) ∧ (r2 = 3) ∧ (r3 = 2) ∧ (2 / 7 * (π * (r1^2 + r2^2 + r3^2)) = (π * 16) * θ + (π * 9) * (1 - θ / (2 * π)) + (π * 4) * θ) → θ = 5 * π / 77 :=
by
  intros,
  sorry

end acute_angle_measure_l411_411149


namespace triangle_orientation_or_coincide_l411_411477

theorem triangle_orientation_or_coincide
  (α β γ : ℝ)
  (A_1 B_1 C_1 A_2 B_2 C_2 : Type)
  [inhabited A_1] [inhabited B_1] [inhabited C_1] [inhabited A_2] [inhabited B_2] [inhabited C_2]
  (triangle_A1B1C1 : α + β + γ = 180)
  (angle_α_ge_120 : α ≥ 120) :
  (A_2 = B_2 ∨ B_2 = C_2 ∨ C_2 = A_2) ∨
  (∃ (P Q R : Type), (inhabited P) ∧ (inhabited Q) ∧ (inhabited R) ∧ (P = Q ∧ Q = R ∧ orientation P Q R = -orientation A_1 B_1 C_1)) :=
  sorry

end triangle_orientation_or_coincide_l411_411477


namespace projection_works_l411_411661

-- Define the given vectors
def v1 : ℝ × ℝ := (2, -3)
def v_proj : ℝ × ℝ := (3, -9 / 2)
def v2 : ℝ × ℝ := (-3, 2)

-- Prove that the projection of v2 onto the vector that projects v1 to v_proj is as given
theorem projection_works :
  let unit_vector : ℝ × ℝ := (1, -1.5) in
  (v2.1 * unit_vector.1 + v2.2 * unit_vector.2) / (unit_vector.1^2 + unit_vector.2^2) * unit_vector = (-1.846, 2.769) :=
by 
  sorry

end projection_works_l411_411661


namespace probability_two_dice_show_1_l411_411575

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411575


namespace unique_square_root_l411_411805

theorem unique_square_root (a x : ℝ) (h1 : sqrt x = 2 * a + 1) (h2 : sqrt x = 4 - a) : x = 1 :=
  sorry

end unique_square_root_l411_411805


namespace triangle_angle_B_leq_60_l411_411858

variable (A B C : Type) [Nonempty (Lt (Angle B))] [IsTriangle A B C]
variable (P : Point) (E : Point)

-- Definitions from conditions
def h_a_is_largest_altitude (h_a : Length) : Prop :=
  h_a = Length (Altitude A) ∧ h_a = max (max (Length (Altitude B)) (Length (Altitude C))) (Length (Altitude A))

def m_b_is_median_equal_h_a (m_b h_a : Length) : Prop :=
  m_b = Length (Median B) ∧ h_a = Length (Median B)

-- Mathematical equivalent proof problem
theorem triangle_angle_B_leq_60
  (h_a : Length) (m_b : Length) 
  (ha_max : h_a_is_largest_altitude A B C h_a)
  (ha_eq_mb : m_b_is_median_equal_h_a A B C m_b h_a)
  : ∠ B ≤ 60 := sorry

end triangle_angle_B_leq_60_l411_411858


namespace james_eats_three_l411_411837

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end james_eats_three_l411_411837


namespace negation_universal_to_existential_l411_411506

theorem negation_universal_to_existential :
  ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_universal_to_existential_l411_411506


namespace highest_probability_two_out_of_three_probability_l411_411734

structure Student :=
  (name : String)
  (P_T : ℚ)  -- Probability of passing the theoretical examination
  (P_S : ℚ)  -- Probability of passing the social practice examination

noncomputable def P_earn (student : Student) : ℚ :=
  student.P_T * student.P_S

def student_A := Student.mk "A" (5 / 6) (1 / 2)
def student_B := Student.mk "B" (4 / 5) (2 / 3)
def student_C := Student.mk "C" (3 / 4) (5 / 6)

theorem highest_probability : 
  P_earn student_C > P_earn student_B ∧ P_earn student_B > P_earn student_A :=
by sorry

theorem two_out_of_three_probability :
  (1 - P_earn student_A) * P_earn student_B * P_earn student_C +
  P_earn student_A * (1 - P_earn student_B) * P_earn student_C +
  P_earn student_A * P_earn student_B * (1 - P_earn student_C) =
  115 / 288 :=
by sorry

end highest_probability_two_out_of_three_probability_l411_411734


namespace ted_candy_bars_l411_411105

theorem ted_candy_bars (b : ℕ) (n : ℕ) (h : b = 5) (h2 : n = 3) : b * n = 15 :=
by
  sorry

end ted_candy_bars_l411_411105


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411542

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411542


namespace parabola_translation_l411_411165

theorem parabola_translation (x : ℝ) : 
  (2, 3 : ℝ) 
    ∧ 
  (∀ x : ℝ, -x^2 + 3 = -((x+2)^2) ) sorry

end parabola_translation_l411_411165


namespace sum_of_possible_values_of_m_eq_neg_two_thirds_l411_411623

theorem sum_of_possible_values_of_m_eq_neg_two_thirds :
  let p := (3 + 2 * x + x^2) * (1 + m * x + m^2 * x^2),
      coeff_x2 := 3 * m^2 + 2 * m + 1
  in coeff_x2 = 1 → m = 0 ∨ m = -2/3 → 0 + (-2/3) = -2/3 :=
by
  sorry

end sum_of_possible_values_of_m_eq_neg_two_thirds_l411_411623


namespace Alina_minus_Polina_eq_1944_l411_411463

-- Define the initial sequence
def seq := [2, 0, 1, 9, 0]

-- Define a function to simulate the transformation
def transform (s : List ℕ) : List ℕ :=
  s.foldr (λ x ys => match ys with
    | []      => [x]
    | y :: ys => x + y :: x :: ys) []

-- Apply 5 transformations
def seq_after_5_steps : List ℕ := (List.replicate 5 transform).foldl (λ s f => f s) seq

-- Define the indices of relevant zeros
def first_zero_index := seq_after_5_steps.index_of 0
def second_zero_index := seq_after_5_steps.drop (first_zero_index + 1).index_of 0 + first_zero_index + 1

-- Define Polina and Alina's sums
def Polina_sum : ℕ := 
  seq_after_5_steps.drop (first_zero_index + 1).take (second_zero_index - first_zero_index - 1).sum

def Alina_sum : ℕ := 
  seq_after_5_steps.sum - Polina_sum

def difference : ℕ := Alina_sum - Polina_sum

-- The statement of the problem
theorem Alina_minus_Polina_eq_1944 : difference = 1944 := by
  sorry

end Alina_minus_Polina_eq_1944_l411_411463


namespace smallest_n_exists_l411_411265

theorem smallest_n_exists (n : ℕ) (h : n ≥ 4) :
  (∃ (S : Finset ℤ), S.card = n ∧
    (∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        (a + b - c - d) % 20 = 0))
  ↔ n = 9 := sorry

end smallest_n_exists_l411_411265


namespace hyperbola_parameters_sum_l411_411396

-- Define the parameters and their given values.
def center : ℝ × ℝ := (-2, 0)
def vertex : ℝ × ℝ := (-7, 0)
def focus : ℝ × ℝ := (-2 + Real.sqrt 41, 0)

-- Define h, k, a, b based on the conditions
def h := -2
def k := 0
def a := abs(h - (-7))  -- Distance from center to vertex
def c := Real.sqrt 41   -- Distance from center to focus
def b := Real.sqrt (c^2 - a^2)  -- Use c^2 = a^2 + b^2

-- The statement to be proven
theorem hyperbola_parameters_sum : h + k + a + b = 7 := by
  sorry

end hyperbola_parameters_sum_l411_411396


namespace rectangle_area_l411_411822

theorem rectangle_area (AB AC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) : ∃ Area : ℝ, Area = 120 :=
by
  sorry

end rectangle_area_l411_411822


namespace prob_two_ones_in_twelve_dice_l411_411568

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411568


namespace min_distance_R_l411_411328

noncomputable def P : ℝ × ℝ := (-2, -3)
noncomputable def Q : ℝ × ℝ := (5, 3)

def line_PQ (x : ℝ) : ℝ := (6 / 7) * x - (9 / 7)

theorem min_distance_R {R : ℝ × ℝ} (h : R = (2, (line_PQ 2))) : (R.2 = 3 / 7) :=
by {
  -- The proof will go here
  sorry
}

end min_distance_R_l411_411328


namespace inequality_proof_l411_411758

theorem inequality_proof (x : ℝ) (n : ℕ) (h : 3 * x ≥ -1) : (1 + x) ^ n ≥ 1 + n * x :=
sorry

end inequality_proof_l411_411758


namespace representation_with_max_terms_and_sum_of_exponents_l411_411187

theorem representation_with_max_terms_and_sum_of_exponents :
  ∃ (S : set ℕ), 1024 = ∑ n in S, 2^n ∧ S = {9, 8, 7, 6, 5, 4} ∧ ∑ n in S, n = 39 :=
by
  sorry

end representation_with_max_terms_and_sum_of_exponents_l411_411187


namespace parabola_angle_mko_45_degrees_l411_411876

open Real

noncomputable def parabola_focus {p : ℝ} (hp : 0 < p): ℝ × ℝ := (p / 2, 0)

noncomputable def parabola_point_m {p : ℝ} (hp : 0 < p): ℝ × ℝ := (p / 2, p)

noncomputable def parabola_directrix_point_k {p : ℝ} (hp : 0 < p): ℝ × ℝ := (-p / 2, 0)

def angle_mko {p : ℝ} (hp : 0 < p) : ℝ :=
  let M := parabola_point_m hp in
  let K := parabola_directrix_point_k hp in
  let slope_km := (M.2 - K.2) / (M.1 - K.1) in
  Real.arctan slope_km * 180 / Real.pi

theorem parabola_angle_mko_45_degrees (p : ℝ) (hp : 0 < p) (h_MF_p : dist (parabola_point_m hp) (parabola_focus hp) = p) :
  angle_mko hp = 45 :=
by
  let M := parabola_point_m hp
  let K := parabola_directrix_point_k hp
  rw [angle_mko, parabola_point_m, parabola_directrix_point_k]
  sorry

end parabola_angle_mko_45_degrees_l411_411876


namespace max_triangles_three_parallel_families_l411_411152

/-- 
Given three families of parallel lines, each containing 10 lines,
the maximum number of triangles that can be formed by these lines is 150.
-/
theorem max_triangles_three_parallel_families (L1 L2 L3 : set (set ℝ)) (h1 : ∀ l ∈ L1, ∀ l' ∈ L1, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y) 
  (h2 : ∀ l ∈ L2, ∀ l' ∈ L2, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y) 
  (h3 : ∀ l ∈ L3, ∀ l' ∈ L3, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y)
  (hL1_card : L1.card = 10) (hL2_card : L2.card = 10) (hL3_card : L3.card = 10) : 
  ∃ max_n : ℕ, max_n = 150 := 
begin
  sorry
end

end max_triangles_three_parallel_families_l411_411152


namespace cross_number_puzzle_digit_star_l411_411193

theorem cross_number_puzzle_digit_star :
  ∃ N₁ N₂ N₃ N₄ : ℕ,
    N₁ % 1000 / 100 = 4 ∧ N₁ % 10 = 1 ∧ ∃ n : ℕ, N₁ = n ^ 2 ∧
    N₃ % 1000 / 100 = 6 ∧ ∃ m : ℕ, N₃ = m ^ 4 ∧
    ∃ p : ℕ, N₂ = 2 * p ^ 5 ∧ 100 ≤ N₂ ∧ N₂ < 1000 ∧
    N₄ % 10 = 5 ∧ ∃ q : ℕ, N₄ = q ^ 3 ∧ 100 ≤ N₄ ∧ N₄ < 1000 ∧
    (N₁ % 10 = 4) :=
by
  sorry

end cross_number_puzzle_digit_star_l411_411193


namespace bread_leftover_after_sandwiches_l411_411091

def total_bread_slices (bread_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  bread_packages * slices_per_package

def total_ham_slices (ham_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  ham_packages * slices_per_package

def sandwiches_from_ham (ham_slices: ℕ) : ℕ :=
  ham_slices

def total_bread_used (sandwiches: ℕ) (bread_slices_per_sandwich: ℕ) : ℕ :=
  sandwiches * bread_slices_per_sandwich

def bread_leftover (total_bread: ℕ) (bread_used: ℕ) : ℕ :=
  total_bread - bread_used

theorem bread_leftover_after_sandwiches :
  let bread_packages := 2
  let bread_slices_per_package := 20
  let ham_packages := 2
  let ham_slices_per_package := 8
  let bread_slices_per_sandwich := 2 in
  bread_leftover
    (total_bread_slices bread_packages bread_slices_per_package)
    (total_bread_used
      (sandwiches_from_ham (total_ham_slices ham_packages ham_slices_per_package))
      bread_slices_per_sandwich) = 8 :=
by
  sorry

end bread_leftover_after_sandwiches_l411_411091


namespace solve_system_eqn_l411_411719

theorem solve_system_eqn :
  ∃ x y : ℚ, 7 * x = -9 - 3 * y ∧ 2 * x = 5 * y - 30 ∧ x = -135 / 41 ∧ y = 192 / 41 :=
by 
  sorry

end solve_system_eqn_l411_411719


namespace TotalToysIsNinetyNine_l411_411362

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l411_411362


namespace inequality_proof_l411_411808

theorem inequality_proof {x y : ℝ} : 
  (2 * x + sqrt (4 * x ^ 2 + 1)) * (sqrt (1 + 4 / y ^ 2) - 2 / y) ≥ 1 → x + y ≥ 2 :=
by
  sorry

end inequality_proof_l411_411808


namespace find_d_l411_411127

theorem find_d (c : ℝ) (d : ℝ) (h1 : c = 7)
  (h2 : (2, 6) ∈ { p : ℝ × ℝ | ∃ d, (p = (2, 6) ∨ p = (5, c) ∨ p = (d, 0)) ∧
           ∃ m, m = (0 - 6) / (d - 2) ∧ m = (c - 6) / (5 - 2) }) : 
  d = -16 :=
by
  sorry

end find_d_l411_411127


namespace sum_of_two_digit_odd_numbers_l411_411517

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end sum_of_two_digit_odd_numbers_l411_411517


namespace simplify_and_rationalize_l411_411886

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411886


namespace last_two_digits_sum_factorials_l411_411967

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l411_411967


namespace min_sum_of_angles_l411_411412

theorem min_sum_of_angles (A B C : ℝ) (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin B + Real.sin C ≤ 1) : 
  min (A + B) (min (B + C) (C + A)) < 30 := 
sorry

end min_sum_of_angles_l411_411412


namespace arithmetic_sequence_third_term_l411_411385

theorem arithmetic_sequence_third_term 
    (a d : ℝ) 
    (h1 : a = 2)
    (h2 : (a + d) + (a + 3 * d) = 10) : 
    a + 2 * d = 5 := 
by
  sorry

end arithmetic_sequence_third_term_l411_411385


namespace Sup_iid_rvs_eq_sup_P_l411_411851

open Probability MeasureTheory

noncomputable theory

variables {Ω : Type*} {ξ ξ₁ ξ₂ : Ω → ℝ} 

-- Assuming ξ, ξ₁, ξ₂, ... are i.i.d. random variables
axiom i.i.d. : ∀ n, (xi : @MeasureTheory.Measure.toOuterMeasure Ω PMeasureSpace.real ℝ) == ξ₁

-- Defining the random variable supremum and x*
def xi_sup : ℝ := ⨆ n, ξ₁
def x_star : ℝ := ⨆ (x : ℝ) (h : PMeasureSpace.real < 1), x

-- Problem statement in Lean 4
theorem Sup_iid_rvs_eq_sup_P {Ω : Type*} [MeasureSpace Ω] [is_probability_measure P] : 
  (i.i.d. ξ₁ ξ₂) → (P ((λ x, xi_sup = x_star) = (1 : ℝ)) :=
by sorry

end Sup_iid_rvs_eq_sup_P_l411_411851


namespace probability_two_ones_in_twelve_dice_l411_411588
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411588


namespace yogurt_cost_l411_411156

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l411_411156


namespace yogurt_production_cost_l411_411161

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l411_411161


namespace product_real_imag_conj_eq_l411_411799

noncomputable def z : ℂ := (3 + -i) / 2
noncomputable def z_conj : ℂ := conj z
noncomputable def real_part : ℝ := z_conj.re
noncomputable def imag_part : ℝ := z_conj.im

theorem product_real_imag_conj_eq : (1 + complex.i) * z = 2 - complex.i →
  real_part * imag_part = 3 / 4 := by
  sorry

end product_real_imag_conj_eq_l411_411799


namespace bob_needs_50_planks_l411_411250

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l411_411250


namespace orange_gumdrops_after_replacement_l411_411653

noncomputable def total_gumdrops : ℕ :=
  100

noncomputable def initial_orange_gumdrops : ℕ :=
  10

noncomputable def initial_blue_gumdrops : ℕ :=
  40

noncomputable def replaced_blue_gumdrops : ℕ :=
  initial_blue_gumdrops / 3

theorem orange_gumdrops_after_replacement : 
  (initial_orange_gumdrops + replaced_blue_gumdrops) = 23 :=
by
  sorry

end orange_gumdrops_after_replacement_l411_411653


namespace an_decreasing_l411_411856

theorem an_decreasing (n : ℕ) (h : 2 ≤ n) :
  let a := λ n, ∑ k in Finset.range n.succ, 1 / (k * (n + 1 - k))
  in a (n + 1) < a n :=
by { sorry }

end an_decreasing_l411_411856


namespace exists_y_with_7_coprimes_less_than_20_l411_411937

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1
def connection (a b : ℕ) : ℚ := Nat.lcm a b / (a * b)

theorem exists_y_with_7_coprimes_less_than_20 :
  ∃ y : ℕ, y < 20 ∧ (∃ x : ℕ, connection y x = 1) ∧ (Nat.totient y = 7) :=
by
  sorry

end exists_y_with_7_coprimes_less_than_20_l411_411937


namespace probability_two_dice_showing_1_l411_411602

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411602


namespace winnie_the_pooh_max_honey_l411_411011

open Real

def pots : Type := Fin 5 → ℝ

noncomputable def sum_honey (p : pots) : ℝ :=
  p 0 + p 1 + p 2 + p 3 + p 4

theorem winnie_the_pooh_max_honey (p : pots) (h : sum_honey p = 3):
  ∃ i : Fin 4, p i + p (i+1) ≥ 1 :=
begin
  sorry
end

end winnie_the_pooh_max_honey_l411_411011


namespace adrian_gets_3_contacts_l411_411676

variable (cost1 : ℝ) (contacts1 : ℝ) (cost2 : ℝ) (contacts2 : ℝ) 

-- Defining conditions for the boxes
def box1 : Prop := cost1 = 25 ∧ contacts1 = 50
def box2 : Prop := cost2 = 33 ∧ contacts2 = 99

-- The cost per contact for each box
noncomputable def cost_per_contact1 : ℝ := contacts1 / cost1
noncomputable def cost_per_contact2 : ℝ := contacts2 / cost2

-- Adrian chooses the box with the smaller cost per contact
def adrian_choice : Prop := (cost_per_contact1 ≤ cost_per_contact2 → cost_per_contact2) ∧ (cost_per_contact2 < cost_per_contact1 → cost_per_contact2)

-- Prove that Adrian will get 3 contacts per dollar in the chosen box
theorem adrian_gets_3_contacts (h1 : box1) (h2 : box2) : 
  adrian_choice cost1 contacts1 cost2 contacts2 → cost_per_contact2 = 3 := 
by
  sorry

end adrian_gets_3_contacts_l411_411676


namespace part1_part2_l411_411199

-- Part 1: define x, y, z > 0, and a, b, c
variables {x y z a b c : ℝ}
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (ha : a = x + 1 / y) (hb : b = y + 1 / z) (hc : c = z + 1 / x)

-- Part 1: Prove that there exists an 'i' in {a, b, c} such that i >= 2
theorem part1 : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

-- Part 2: define a, b, c as sides of triangle ABC
variables {A B C : Point}
variables (triangle_ABC : Triangle A B C)
variables (ha : a = side_length A B)
variables (hb : b = side_length B C)
variables (hc : c = side_length C A)

-- Part 2: Prove that (a + b) / (1 + a + b) > c / (1 + c)
theorem part2 : (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end part1_part2_l411_411199


namespace bus_stops_for_40_minutes_per_hour_l411_411716

theorem bus_stops_for_40_minutes_per_hour 
  (speed_excluding_stoppages : ℕ)
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 20) : 
  (time_stopped : ℕ) := 
  by
  have speed_loss := speed_excluding_stoppages - speed_including_stoppages,
  have ratio := (speed_loss : ℤ) / speed_excluding_stoppages,
  have minutes_per_hour_stopped := ratio * 60,
  have time_stopped = minutes_per_hour_stopped
  rw [h1, h2] at *,
  sorry

end bus_stops_for_40_minutes_per_hour_l411_411716


namespace mangoes_total_l411_411683

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l411_411683


namespace problem1_second_to_last_term_problem2_binomial_coeff_sum_l411_411275

-- Definitions for the binomial term and its general term
noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (nat.choose n r) * (real.sqrt x)^(n - r) * ((-2 / x)^r)

-- Problem (1): Prove the second to last term when n = 6
theorem problem1_second_to_last_term (x : ℝ) (hx : x ≠ 0) : binomial_term 6 5 x = -192 * x^(-9 / 2) :=
by sorry

-- Problem (2): Prove the sum of binomial coefficients is 1024 given the coefficient ratio condition
theorem problem2_binomial_coeff_sum (n : ℕ) 
  (h : (16 * nat.choose n 4) / (4 * nat.choose n 2) = 56 / 3) 
  : (finset.range (n + 1)).sum (λ k, (nat.choose n k)) = 1024 :=
by sorry

end problem1_second_to_last_term_problem2_binomial_coeff_sum_l411_411275


namespace problem_statement_l411_411790

noncomputable def g (x : ℝ) : ℝ := 3^(x + 1)

theorem problem_statement (x : ℝ) : g (x + 1) - 2 * g x = g x := by
  -- The proof here is omitted
  sorry

end problem_statement_l411_411790


namespace UF_championship_ratio_l411_411616

theorem UF_championship_ratio 
  (total_points_prev_games : ℕ) 
  (num_prev_games : ℕ) 
  (opp_points_champ : ℕ) 
  (opp_lost_by : ℕ) 
  (h1 : total_points_prev_games = 720)
  (h2 : num_prev_games = 24)
  (h3 : opp_points_champ = 11)
  (h4 : opp_lost_by = 2) :
  (13 : ℚ) / (30 : ℚ) = (11 + 2 : ℕ) / (720 / 24 : ℚ) :=
by {
  norm_num at *,
  sorry
}

end UF_championship_ratio_l411_411616


namespace surface_area_of_tetrahedron_l411_411524

-- Define the necessary prerequisites: Edge length, sin(60 degrees)
def a : ℝ := sorry

lemma sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := sorry

-- Define the formula for the area of an equilateral triangle
def area_equilateral_triangle (a : ℝ) : ℝ :=
  (1 / 2) * a * a * (Real.sqrt 3 / 2)

-- Define the surface area of a tetrahedron
def surface_area_tetrahedron (a : ℝ) : ℝ :=
  4 * area_equilateral_triangle a

-- The theorem to be proved
theorem surface_area_of_tetrahedron (a : ℝ) : surface_area_tetrahedron a = Real.sqrt 3 * a^2 :=
by
  sorry

end surface_area_of_tetrahedron_l411_411524


namespace tetrahedron_plane_inequality_l411_411660

theorem tetrahedron_plane_inequality
  (A B C D K L M N : Point)
  (tetrahedron : Tetrahedron A B C D)
  (P : Plane)
  (intersections : Intersects P (Edge A B) K ∧ Intersects P (Edge B C) L ∧ 
    Intersects P (Edge C D) M ∧ Intersects P (Edge A D) N) :
  (dist A K / dist A B) * (dist B L / dist B C) * (dist C M / dist C D) * (dist D N / dist A D) ≤ 
    1 / 16 := 
sorry

end tetrahedron_plane_inequality_l411_411660


namespace polynomial_remainder_division_l411_411138

theorem polynomial_remainder_division :
  ∀ (x : ℝ), ∃ q r : ℝ[x], 
  (r.degree < (polynomial.C 1 * (polynomial.X^2 - 1)).degree) ∧ 
  (x^12 - x^6 + 1 = q * (polynomial.X^2 - 1) + r) ∧ 
  r = 1 := 
by
  sorry

end polynomial_remainder_division_l411_411138


namespace yogurt_production_cost_l411_411162

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l411_411162


namespace team_red_cards_l411_411018

-- Define the conditions
def team_size (n : ℕ) := n = 11
def players_without_caution (n : ℕ) := n = 5
def yellow_cards_per_cautioned_player := 1
def yellow_cards_per_red_card := 2

-- Define the proof statement
theorem team_red_cards : 
  ∀ (total_players cautioned_players yellow_cards total_red_cards : ℕ),
  team_size total_players →
  players_without_caution (total_players - cautioned_players) →
  cautioned_players * yellow_cards_per_cautioned_player = yellow_cards →
  yellow_cards / yellow_cards_per_red_card = total_red_cards →
  total_red_cards = 3 :=
by
  intros total_players cautioned_players yellow_cards total_red_cards
  assume h1 h2 h3 h4
  sorry

end team_red_cards_l411_411018


namespace final_coordinates_l411_411430

def theta (n : ℕ) (h : n ≥ 2) : ℝ :=
  2 * Real.pi / n

def P_k (k n : ℕ) (h : n ≥ 2) : ℝ × ℝ :=
  (k, 0)

def R_k (k n : ℕ) (h : n ≥ 2) (z : ℂ) : ℂ :=
  let ω := Complex.exp (Complex.I * theta n h)
  ω * (z - k + 0 * Complex.I) + k

def R_seq (n : ℕ) (h : n ≥ 2) (z : ℂ) : ℂ :=
  (List.range n).foldr (λ k, R_k k.succ n h) z

theorem final_coordinates (n : ℕ) (h : n ≥ 2) (x y : ℝ) :
  R_seq n h (Complex.mk x y) = Complex.mk (x + n) y :=
sorry

end final_coordinates_l411_411430


namespace number_of_planks_needed_l411_411251

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l411_411251


namespace Blake_initial_amount_l411_411248

theorem Blake_initial_amount (half_sale_amount : ℕ) (tripled_value_fraction : ℕ) (initial_amount_fraction : ℕ) :
  (half_sale_amount = 30000) ∧ (tripled_value_fraction = 3) ∧ (initial_amount_fraction = 2) →
  let initial_amount := (half_sale_amount * initial_amount_fraction) / tripled_value_fraction in
  initial_amount = 20000 :=
by
  sorry

end Blake_initial_amount_l411_411248


namespace probability_two_dice_show_1_l411_411574

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411574


namespace positive_difference_l411_411842

def Jo_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def round_down_sum (n : ℕ) : ℕ :=
  let multiples := (list.range n).map (λ x, (x / 10) * 10)
  multiples.sum

theorem positive_difference (n : ℕ) (hn : n = 100) :
  |Jo_sum n - round_down_sum n| = 450 :=
by {
  sorry  -- the proof would go here
}

end positive_difference_l411_411842


namespace parabola_unique_solution_l411_411154

theorem parabola_unique_solution (a : ℝ) :
  (∃ x : ℝ, (0 ≤ x^2 + a * x + 5) ∧ (x^2 + a * x + 5 ≤ 4)) → (a = 2 ∨ a = -2) :=
by
  sorry

end parabola_unique_solution_l411_411154


namespace vertex_of_quadratic_l411_411114

def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem vertex_of_quadratic :
  let y := -1 * (x + 1)^2 - 8
  (quadratic_vertex (-1) 2 7) = (-1, -8) :=
sorry

end vertex_of_quadratic_l411_411114


namespace tiles_on_board_eq_distances_l411_411269

noncomputable def distance (x y : ℕ) : ℝ := real.sqrt (x ^ 2 + y ^ 2)

theorem tiles_on_board_eq_distances :
  ∃ (p1 p2 p3 p4 : ℕ), p1 ≠ p3 ∧ p2 ≠ p4 ∧
  ( ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 ∧ i ≠ j → distance i j = distance p1 p2 → distance p1 p2 = distance p3 p4) :=
begin
  sorry
end

end tiles_on_board_eq_distances_l411_411269


namespace increasing_branch_of_inverse_proportion_l411_411024

-- Define the inverse proportion function and the conditions
def inverse_proportion (k x : ℝ) : ℝ := k / x

-- The Lean theorem statement
theorem increasing_branch_of_inverse_proportion (k : ℝ) (h_knz : k ≠ 0) (h_neg : k < 0) :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ inverse_proportion k x1 < inverse_proportion k x2 :=
sorry

end increasing_branch_of_inverse_proportion_l411_411024


namespace two_digit_product_GCD_l411_411929

-- We define the condition for two-digit integer numbers
def two_digit_num (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Lean statement capturing the conditions
theorem two_digit_product_GCD :
  ∃ (a b : ℕ), two_digit_num a ∧ two_digit_num b ∧ a * b = 1728 ∧ Nat.gcd a b = 12 := 
by {
  sorry -- The proof steps would go here
}

end two_digit_product_GCD_l411_411929


namespace michael_scoops_l411_411866

def needs_flour_cups : ℝ := 6
def total_flour_cups : ℝ := 8
def measuring_cup_size : ℝ := 1 / 4

theorem michael_scoops : (total_flour_cups - needs_flour_cups) / measuring_cup_size = 8 := 
  by
  sorry

end michael_scoops_l411_411866


namespace distance_point_to_line_l411_411350

-- Definitions based on the given parametric equation for the line
def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  (1 + 3 * t, 2 + 4 * t)

-- Standard form of the line equation derived from the parametric equations
def A : ℝ := 4
def B : ℝ := -3
def C : ℝ := 2

-- Given point (x1, y1)
def x1 : ℝ := 1
def y1 : ℝ := 0

-- Definition for the distance formula
def distance (A B C x1 y1 : ℝ) : ℝ :=
  abs (A * x1 + B * y1 + C) / (Real.sqrt (A * A + B * B))

-- Main theorem stating the equivalent problem
theorem distance_point_to_line :
  distance A B C x1 y1 = 6 / 5 := 
by 
  sorry

end distance_point_to_line_l411_411350


namespace marked_price_l411_411216

theorem marked_price (initial_price discount_rate cost_profit_rate marked_discount_rate final_marked_price : ℝ)
  (h_initial_price : initial_price = 30)
  (h_discount_rate : discount_rate = 0.10)
  (h_cost_profit_rate : cost_profit_rate = 0.25)
  (h_marked_discount_rate : marked_discount_rate = 0.15)
  (h_final_marked_price : final_marked_price = 39.70) :
  let purchase_price := initial_price * (1 - discount_rate) in
  let desired_selling_price := purchase_price * (1 + cost_profit_rate) in
  let marked_price := desired_selling_price / (1 - marked_discount_rate) in
  marked_price = final_marked_price :=
by
  sorry

end marked_price_l411_411216


namespace bases_final_digit_625_is_1_l411_411305

theorem bases_final_digit_625_is_1 : 
  {b : ℕ | 2 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0}.card = 5 := 
by
  sorry

end bases_final_digit_625_is_1_l411_411305


namespace probability_two_ones_in_twelve_dice_l411_411553

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411553


namespace total_big_cats_l411_411245

theorem total_big_cats : 
  let lions := 12
  let tigers := 14
  let combined := lions + tigers
  let cougars := combined / 3
  in lions + tigers + Int.floor cougars = 34 :=
by
  let lions := 12
  let tigers := 14
  let combined := lions + tigers
  let cougars := combined / 3
  have : lions + tigers + Int.floor cougars = 34 := sorry
  exact this

end total_big_cats_l411_411245


namespace last_two_digits_sum_factorials_l411_411969

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l411_411969


namespace find_m_interval_l411_411206

noncomputable def is_even_function (g : ℝ → ℝ) : Prop :=
∀ x, g x = g (-x)

noncomputable def is_monotonically_increasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ y → y ≤ b → g x ≤ g y

theorem find_m_interval
  (g : ℝ → ℝ)
  (h_even : is_even_function g)
  (h_mono : is_monotonically_increasing g 0 3)
  (h_ineq : ∀ m, g (1 - m) < g m):
  ∀ m, m ∈ Ioc (1/2 : ℝ) 3 :=
by
  sorry

end find_m_interval_l411_411206


namespace net_sale_effect_l411_411637

theorem net_sale_effect (P S : ℝ) : 
    let original_revenue := P * S in
    let new_revenue := 0.8 * P * (1.8 * S) in
    (new_revenue - original_revenue) / original_revenue = 0.44 :=
by
    sorry

end net_sale_effect_l411_411637


namespace pyramid_surface_area_and_inscribed_sphere_radius_l411_411407

noncomputable section

-- Define basic conditions
variables (A B C S : Point)
variables (AB AC : ℝ) (BC : ℝ) (SB_height : ℝ)

-- Assume the given conditions
def pyramid_conditions : Prop :=
  AB = 10 ∧ AC = 10 ∧ BC = 16 ∧ SB_height = 4

-- Define the pyramid
def is_pyramid (Q : ℝ) (r : ℝ) : Prop :=
  (Q = 152) ∧ (r = 24 / 19)

theorem pyramid_surface_area_and_inscribed_sphere_radius :
  pyramid_conditions A B C S AB AC BC SB_height → ∃ Q r, is_pyramid Q r :=
by
  sorry

end pyramid_surface_area_and_inscribed_sphere_radius_l411_411407


namespace min_value_arithmetic_sequence_l411_411751

theorem min_value_arithmetic_sequence (d : ℝ) (n : ℕ) (hd : d ≠ 0) (a1 : ℝ) (ha1 : a1 = 1)
(geo : (1 + 2 * d)^2 = 1 + 12 * d) (Sn : ℝ) (hSn : Sn = n^2) (an : ℝ) (han : an = 2 * n - 1) :
  ∀ (n : ℕ), n > 0 → (2 * Sn + 8) / (an + 3) ≥ 5 / 2 :=
by sorry

end min_value_arithmetic_sequence_l411_411751


namespace geometric_sequence_value_l411_411392

variable {a : ℕ → ℝ}
variable {a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r -- generic geometric sequence def, though it should adapt to specific elements if needed

noncomputable def a_eq_mul : Prop :=
  a 3 * a 4 * a 5 = 3 ∧ a 6 * a 7 * a 8 = 21

theorem geometric_sequence_value :
  a_eq_mul →
  a 9 * a 10 * a 11 = 147 := by
  sorry

end geometric_sequence_value_l411_411392


namespace probability_two_ones_in_twelve_dice_l411_411547

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411547


namespace super_cool_rectangle_area_sum_l411_411663

theorem super_cool_rectangle_area_sum :
  (∑ (area : ℕ) in {area : ℕ | ∃ (a b : ℕ), a * b = 3 * (a + b) ∧ a ≠ b}, area)
  + (∃ (a b : ℕ), a = b ∧ (a - 3) * (b - 3) = 9) ∑ (area : ℕ) in {a * a}, a * a = 84 :=
sorry

end super_cool_rectangle_area_sum_l411_411663


namespace pentagon_C_y_coordinate_l411_411393

/--
In a pentagon ABCDE, there is a vertical line of symmetry.
Vertex E is moved to (5,0), while A(0,0), B(0,5), and D(5,5)
Prove that the y-coordinate of vertex C is 21 such that the area of pentagon ABCDE becomes 65 square units.
-/
theorem pentagon_C_y_coordinate
  (E : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ)
  (h : ℝ) :
  E = (5,0) ∧ A = (0,0) ∧ B = (0,5) ∧ D = (5,5) ∧
  (∃ C : ℝ × ℝ, C = (5,h)) ∧
  let square_area := 25 in
  let pentagon_area := 65 in
  let triangle_area := pentagon_area - square_area in
  (triangle_area = 40) ∧
  (1 / 2 * 5 * (h - 5) = 40)
  → h = 21 :=
sorry

end pentagon_C_y_coordinate_l411_411393


namespace maximum_candies_eaten_l411_411145

theorem maximum_candies_eaten (erase_sum_candies :
  ∀ (n : ℕ) (x y : ℕ), 0 < n → x + y = n → x * y → ℕ) :
  (∑ i in finset.range 46, 1) * 45 = 1035 :=
begin
  -- We start with sum of 46 ones, which is 46.
  have h : ∑ i in finset.range 46, 1 = 46,
    from finset.sum_const_nat 1 (finset.range 46).card (by norm_num[
    finset.range_46, finset.card_range]),
  -- Reduce the problem to the product of pairs of erased numbers over 45 minutes.
  have h1 : (46) * 45 = 1035,
    from calc 46 * 45 = 1035 : by norm_num,
  exact h1,
end

end maximum_candies_eaten_l411_411145


namespace probability_two_dice_showing_1_l411_411607

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411607


namespace proof_inequality_l411_411381

theorem proof_inequality (p q r : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hpr_lt_qr : p * r < q * r) : 
  p < q :=
by 
  sorry

end proof_inequality_l411_411381


namespace chord_length_proof_l411_411488

noncomputable def chord_length : ℝ :=
  let line_eq (x y : ℝ) := 3 * x - 4 * y - 4 = 0
  let circle_eq (x y : ℝ) := (x - 3) * (x - 3) + y * y = 9
  4 * real.sqrt 2

theorem chord_length_proof :
  ∀ (x y : ℝ), (3 * x - 4 * y - 4 = 0) →
  ((x - 3) * (x - 3) + y * y = 9) →
  chord_length = 4 * real.sqrt 2 :=
by
  sorry

end chord_length_proof_l411_411488


namespace coeff_x3_in_expansion_l411_411916

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

open Polynomial

theorem coeff_x3_in_expansion (x : ℚ) :
  coeff (expand (1 - 2 * polynomial.X) (* Expand (1 - x) to the 5th power *)
  (expand (1 - polynomial.X) ^ 5)) 3 = -30 := 
sorry

end coeff_x3_in_expansion_l411_411916


namespace difference_between_two_numbers_l411_411612

theorem difference_between_two_numbers (a b : ℕ) (h₁ : a + b = 34) (h₂ : max a b = 22) : |a - b| = 10 := by
  sorry

end difference_between_two_numbers_l411_411612


namespace there_exists_triangle_part_two_l411_411062

noncomputable def exists_triangle (a b c : ℝ) : Prop :=
a > 0 ∧
4 * a - 8 * b + 4 * c ≥ 0 ∧
9 * a - 12 * b + 4 * c ≥ 0 ∧
2 * a ≤ 2 * b ∧
2 * b ≤ 3 * a ∧
b^2 ≥ a*c

theorem there_exists_triangle (a b c : ℝ) (h1 : a > 0)
  (h2 : 4 * a - 8 * b + 4 * c ≥ 0)
  (h3 : 9 * a - 12 * b + 4 * c ≥ 0)
  (h4 : 2 * a ≤ 2 * b)
  (h5 : 2 * b ≤ 3 * a)
  (h6 : b^2 ≥ a * c) : 
 a ≤ b ∧ b ≤ c ∧ a + b > c :=
sorry

theorem part_two (a b c : ℝ) (h1 : a > 0) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < a + b) :
  ∃ h : a > 0, (a / (a + c) + b / (b + a) > c / (b + c)) :=
sorry

end there_exists_triangle_part_two_l411_411062


namespace largest_non_prime_sum_l411_411618

theorem largest_non_prime_sum (a b n : ℕ) (h1 : a ≥ 1) (h2 : b < 47) (h3 : n = 47 * a + b) (h4 : ∀ b, b < 47 → ¬Nat.Prime b → b = 43) : 
  n = 90 :=
by
  sorry

end largest_non_prime_sum_l411_411618


namespace distance_and_area_of_triangle_l411_411283

theorem distance_and_area_of_triangle :
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  distance = 10 ∧ area = 24 :=
by
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  have h_dist : distance = 10 := sorry
  have h_area : area = 24 := sorry
  exact ⟨h_dist, h_area⟩

end distance_and_area_of_triangle_l411_411283


namespace minimum_positive_temperatures_l411_411244
-- Importing the relevant libraries

-- Defining the participants' totals and conditions to match the given problem
def numberOfParticipants := 9
def recordedPositiveNumbers := 36
def recordedNegativeNumbers := 36

-- Mathematically equivalent proof problem in Lean 4
theorem minimum_positive_temperatures (x y : ℕ) (hx : x = numberOfParticipants) 
  (hpos : recordedPositiveNumbers + recordedNegativeNumbers = x * (x - 1)) 
  (hprod : recordedPositiveNumbers = y * (y - 1) + (x - y) * (x - y - 1)) :
  y ≥ 3 :=
begin
  sorry -- Proof to be filled in later
end

end minimum_positive_temperatures_l411_411244


namespace snail_no_crossing_twice_l411_411464

-- Define the problem conditions
-- Define a type for a point on the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line on the plane (given as Ax + By = C)
structure Line :=
  (A B C : ℝ)

-- 2017 lines in the plane
constant lines : List Line
constant num_lines : lines.length = 2017

-- No three lines are concurrent
axiom no_three_concurrent (l1 l2 l3 : Line) :
  ¬(∃ (p : Point), 
    (l1.A * p.x + l1.B * p.y = l1.C) ∧
    (l2.A * p.x + l2.B * p.y = l2.C) ∧
    (l3.A * p.x + l3.B * p.y = l3.C)
  )

-- A snail departs from a non-intersection point on one of these lines
constant start_line : Line
constant start_point : Point
axiom start_point_on_line : start_line.A * start_point.x + start_line.B * start_point.y = start_line.C
axiom start_point_not_intersection : ¬(∃ (l : Line), (l ≠ start_line) ∧ (l.A * start_point.x + l.B * start_point.y = l.C))

-- The snail moves according to the specified rules
-- (details omitted as per the problem description, only final theorem needed)

-- Theorem statement: No line segment is crossed in both directions
theorem snail_no_crossing_twice :
  ¬(∃ (l : Line), 
    (∃ (p1 p2 : Point), 
      (l.A * p1.x + l.B * p1.y = l.C) ∧ 
      (l.A * p2.x + l.B * p2.y = l.C) ∧ 
      p1 ≠ p2 ∧
      -- Dummy predicate for traversal, actual detailed logic omitted for simplicity
      (snail_crosses_both_directions l p1 p2) )
  ) := 
  sorry -- Proof omitted

end snail_no_crossing_twice_l411_411464


namespace domain_real_iff_m_nonneg_l411_411346

theorem domain_real_iff_m_nonneg (m : ℝ) : 
  (∀ x : ℝ, mx^2 - 6mx + 9m + 8 ≥ 0) ↔ (0 ≤ m) := sorry

end domain_real_iff_m_nonneg_l411_411346


namespace problem_1_problem_2_l411_411338

noncomputable def f (x : ℝ) (A ω : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 3)

theorem problem_1 (A ω : ℝ) 
  (A_pos : A > 0) 
  (ω_pos : ω > 0) 
  (max_value : ∀ x, f x A ω ≤ 2)
  (period : ∀ x, f x A ω = f (x + Real.pi) A ω) : A = 2 ∧ ω = 2 := sorry

theorem problem_2 (A ω : ℝ)
  (A_pos : A > 0)
  (ω_pos : ω > 0)
  (fAω : ∀ x, f x A ω = A * Real.sin (ω * x + Real.pi / 3))
  (A_eq_2 : A = 2)
  (ω_eq_2 : ω = 2) :
  Set.range (f x A ω ∣ x ∈ (Set.Icc 0 (Real.pi / 2))) = Set.Icc (- Real.sqrt 3) 2 := sorry

end problem_1_problem_2_l411_411338


namespace gcd_lcm_product_l411_411293

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_product (a b : ℕ) : gcd a b * lcm a b = a * b :=
begin
  apply Nat.gcd_mul_lcm,
end

example : gcd 12 9 * lcm 12 9 = 108 :=
by
  have h : gcd 12 9 * lcm 12 9 = 12 * 9 := gcd_lcm_product 12 9
  rw [show 12 * 9 = 108, by norm_num] at h
  exact h

end gcd_lcm_product_l411_411293


namespace largest_divisor_of_composite_sum_and_square_l411_411732

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_divisor_of_composite_sum_and_square (n : ℕ) (h : is_composite n) : ( ∃ (k : ℕ), ∀ n : ℕ, is_composite n → ∃ m : ℕ, n + n^2 = m * k) → k = 2 :=
by
  sorry

end largest_divisor_of_composite_sum_and_square_l411_411732


namespace minimum_value_function_l411_411497

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem minimum_value_function : ∃ x : ℝ, x ∈ set.Icc (-2 : ℝ) (2 : ℝ) ∧ f x = -37 :=
by 
  -- Lean requires a complete proof to confirm ∃ x, x ∈ [-2, 2] and f x = -37
  sorry

end minimum_value_function_l411_411497


namespace companies_employees_combined_in_january_l411_411256

theorem companies_employees_combined_in_january :
  (let P_jan := 500 / 1.15 in 
   let Q_jan := 550 / 1.10 in 
   let R_jan := 600 / 1.20 in 
   P_jan + Q_jan + R_jan = 1435) := 
begin
  sorry
end

end companies_employees_combined_in_january_l411_411256


namespace solve_for_x_l411_411791

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 7 = 13) : x = 42 :=
sorry

end solve_for_x_l411_411791


namespace approximation_example1_approximation_example2_approximation_example3_l411_411186

theorem approximation_example1 (α β : ℝ) (hα : α = 0.0023) (hβ : β = 0.0057) :
  (1 + α) * (1 + β) = 1.008 := sorry

theorem approximation_example2 (α β : ℝ) (hα : α = 0.05) (hβ : β = -0.03) :
  (1 + α) * (10 + β) = 10.02 := sorry

theorem approximation_example3 (α β γ : ℝ) (hα : α = 0.03) (hβ : β = -0.01) (hγ : γ = -0.02) :
  (1 + α) * (1 + β) * (1 + γ) = 1 := sorry

end approximation_example1_approximation_example2_approximation_example3_l411_411186


namespace minimum_value_c_l411_411195

-- Define the problem statement in Lean

variable {V : Type} [inner_product_space ℝ V] [nontrivial V]
variable (OA OB OC K1 K2 : V)
variable (r : ℝ)

noncomputable def c_min : ℝ := 4/3

def non_collinear (u v : V) : Prop := ∃ (a b : ℝ), a * u + b * v ≠ 0 ∧ u ≠ 0 ∧ v ≠ 0

def overrightarrow_OC (OA OB : V) (r : ℝ) : V :=
  (1 / (1 + r)) • OA + (r / (1 + r)) • OB

def M (K : V) (OA OB OC : V) :=
  ∃ (r : ℝ), ∀ K1 K2 ∈ M, (r ≥ 2) ∧ 
  (inner (K - OA) (K - OC) / ∥K - OA∥ = inner (K - OB) (K - OC) / ∥K - OB∥)

theorem minimum_value_c 
  (h1 : non_collinear OA OB)
  (h2 : OC = overrightarrow_OC OA OB r)
  (h3 : ∀ K1 K2 ∈ M K OA OB OC, r ≥ 2 → ∥K1 - K2∥ ≤ c_min * ∥OA - OB∥) :
  ∃ c : ℝ, c = c_min 
  sorry

end minimum_value_c_l411_411195


namespace current_page_l411_411047

variable (Jo : Type → Type)
variable (P : Nat) -- current page
variable (R : Nat) -- reading rate in pages per hour

-- Given conditions
def steady_pace (t1 t2 : Nat) : Prop := (P = t1 * R + t2 * R)
def book_has_210_pages : Prop := (P + 4 * R = 210) -- will read 4 more hours
def an_hour_ago_page_60 : Prop := (P = 60 + R)

theorem current_page (h1 : steady_pace (P - 60) R) (h2 : book_has_210_pages) (h3 : an_hour_ago_page_60) : P = 90 := by
  sorry

end current_page_l411_411047


namespace sqrt_eq_ten_then_x_eq_31_l411_411729

theorem sqrt_eq_ten_then_x_eq_31 (x : ℝ) (h : sqrt (3 * x + 7) = 10) : x = 31 :=
sorry

end sqrt_eq_ten_then_x_eq_31_l411_411729


namespace equilateral_triangle_side_length_l411_411877

theorem equilateral_triangle_side_length : 
  ∀ (A B : ℝ × ℝ), 
    (∃ a1 a2 : ℝ, A = (a1, a2) ∧ B = (-a1, a2) ∧ a2 = -1/2 * a1^2) ∧
    (∃ A1 A2 : ℝ, B = (A1, A2) ∧ A2 = -1/2 * A1^2) ∧
    (A = B ∧ O = (0,0))
    → ∃ l : ℝ, l = 4 * real.sqrt 3 :=
begin
  sorry
end

end equilateral_triangle_side_length_l411_411877


namespace compute_P_3_neg_3_l411_411414

noncomputable def P (m n : ℤ) : ℤ :=
  if m > 0 ∧ n > 0 then
    ∑ i in finset.range m, ∑ j in finset.range n, (i + j + 2 : ℤ) ^ 7
  else
    0

theorem compute_P_3_neg_3 : P 3 (-3) = -2445 := by
  sorry

end compute_P_3_neg_3_l411_411414


namespace distance_P1_P2_is_sqrt3_l411_411023

-- Define the points P₁, P, and P₂
def P1 : ℝ³ := (2, 4, 6)
def P  : ℝ³ := (1, 3, -5)
def P2 : ℝ³ := (1, 3, 5) -- symmetric point of P about xOy

-- Define the distance function
noncomputable def distance (a b : ℝ³) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

-- Proof statement
theorem distance_P1_P2_is_sqrt3 : distance P1 P2 = real.sqrt 3 :=
by
  sorry

end distance_P1_P2_is_sqrt3_l411_411023


namespace problem_1_min_max_problem_2_monotonic_l411_411343

-- Define the function
def f (x : ℝ) (a : ℝ) := x^2 + 2 * a * x + 2

-- Define the interval
def interval := set.Icc (-5 : ℝ) 5

-- Question 1: Prove the minimum and maximum values for a = -1
theorem problem_1_min_max : 
  (∀ x ∈ interval, f x (-1) ≥ 1) ∧ (∃ x ∈ interval, f x (-1) = 1) ∧
  (∀ x ∈ interval, f x (-1) ≤ 37) ∧ (∃ x ∈ interval, f x (-1) = 37) :=
  sorry

-- Question 2: Prove the range of a for monotonicity
theorem problem_2_monotonic (a : ℝ) : 
  (∀ x1 x2 ∈ interval, x1 ≤ x2 → f x1 a ≤ f x2 a) ↔ (a ≤ -5 ∨ a ≥ 5) :=
  sorry

end problem_1_min_max_problem_2_monotonic_l411_411343


namespace last_two_digits_of_sum_l411_411958

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l411_411958


namespace distance_from_point_A_to_line_l_l411_411117

-- Define the point A as a tuple
def A : ℝ × ℝ := (2, 0)

-- Define the line l as an equation
def line (x y : ℝ) : Prop := y = x + 2

-- Define the distance formula from a point to a line
def distance_from_point_to_line (A : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : ℝ :=
  let (x1, y1) := A
  let A := 1
  let B := -1
  let C := 2
  (abs (A * x1 + B * y1 + C)) / (sqrt (A^2 + B^2))

-- Comment explaining fetching the distance
-- This comment replaces the solution steps
-- Calculate the distance from point A to the line l and verify it equals 2sqrt(2)
theorem distance_from_point_A_to_line_l : distance_from_point_to_line A line = 2 * sqrt 2 :=
  sorry

end distance_from_point_A_to_line_l_l411_411117


namespace fibonacci_polynomial_identity_l411_411850

open Nat Polynomial

def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_polynomial_identity (P : Polynomial ℕ) (n : ℕ)
  (hP : ∀ k ∈ Icc (n + 2) (2 * n + 2), P.eval k = fibonacci k) :
  P.eval (2 * n + 3) = fibonacci (2 * n + 3) - 1 :=
begin
  sorry
end

end fibonacci_polynomial_identity_l411_411850


namespace trigonometric_identity_l411_411333

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) * Real.cos (Real.pi / 4 - α) = 7 / 18 :=
by sorry

end trigonometric_identity_l411_411333


namespace quadrilateral_circumcircle_diameter_l411_411614

theorem quadrilateral_circumcircle_diameter :
  ∀ (a b c d e f: ℝ), 
    a = 15 → b = 20 → c = 25 → d = 30 → e = 40 → f = 50 →
    a^2 + b^2 = c^2 → d^2 + e^2 = f^2 →
    ∃ (diameter: ℝ), diameter = 50 :=
by
  intros a b c d e f ha hb hc hd he hf h1 h2
  have h3 : c = 25 := hc
  have h4 : f = 50 := hf
  use 50
  exact h4

end quadrilateral_circumcircle_diameter_l411_411614


namespace segments_intersect_l411_411394

theorem segments_intersect (points : Finset (ℝ × ℝ)) (segments : Finset (Finset (ℝ × ℝ))) : 
  points.card = 35 → 
  segments.card = 100 → 
  (∀ p1 p2 p3 ∈ points, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬Collinear ℝ {p1, p2, p3}) →
  ∃ s1 s2 ∈ segments, s1 ≠ s2 ∧ s1 ∩ s2 ≠ ∅ :=
by
  sorry

end segments_intersect_l411_411394


namespace arithmetic_sequence_satisfies_condition_l411_411624

theorem arithmetic_sequence_satisfies_condition (n : ℕ) :
  let a1 := 4
      d := 8
      an := 8 * n - 4
      Sn := n / 2 * (a1 + an)
  in Sn = 4 * (n ^ 2) := 
by
  sorry

end arithmetic_sequence_satisfies_condition_l411_411624


namespace parabola_intersection_difference_l411_411130

theorem parabola_intersection_difference :
  let intersect_xs := {x | 3 * x^2 - 6 * x + 6 = -2 * x^2 - 4 * x + 6}
  in ∃ a c, a ∈ intersect_xs ∧ c ∈ intersect_xs ∧ c > a ∧ c - a = 2 / 5 :=
by
  sorry

end parabola_intersection_difference_l411_411130


namespace max_knights_seated_l411_411076

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l411_411076


namespace max_bishops_on_chessboard_l411_411620

-- Define the chessboard size
def chessboard_size : ℕ := 8

-- Define the maximum bishops per diagonal
def max_bishops_per_diagonal : ℕ := 3

-- Define the solution for the maximum number of bishops
theorem max_bishops_on_chessboard (n : ℕ) (max_bishops : ℕ) : n = 8 → max_bishops = 3 → (∃ b : ℕ, b = 38) :=
by {
  intros hn hmax,
  use 38,
  sorry
}

end max_bishops_on_chessboard_l411_411620


namespace set_of_positive_numbers_set_of_fractions_set_of_integers_l411_411717

-- Define the given list of numbers
def numbers : List Real := [-3/5, 8, 0, -0.3, -100, Real.pi, 2.1010010001]

-- Define what a positive number is
def is_positive (x : Real) : Prop := x > 0

-- Define what a fraction is (assuming fractions include negative values)
def is_fraction (x : Real) : Prop := ∃ a b : Int, b ≠ 0 ∧ x = a / b

-- Define what an integer is
def is_int (x : Real) : Prop := ∃ a : Int, x = a

-- Problem statements to prove
theorem set_of_positive_numbers :
  {x ∈ numbers | is_positive x} = {8, Real.pi, 2.1010010001} := sorry

theorem set_of_fractions :
  {x ∈ numbers | is_fraction x} = {-3/5, -0.3} := sorry

theorem set_of_integers :
  {x ∈ numbers | is_int x} = {8, 0, -100} := sorry

end set_of_positive_numbers_set_of_fractions_set_of_integers_l411_411717


namespace ratio_of_perimeters_l411_411485

theorem ratio_of_perimeters (A1 A2 : ℝ) (h : A1 / A2 = 16 / 81) : 
  let s1 := real.sqrt A1 
  let s2 := real.sqrt A2 
  (4 * s1) / (4 * s2) = 4 / 9 :=
by {
  sorry
}

end ratio_of_perimeters_l411_411485


namespace distance_home_to_school_l411_411189

theorem distance_home_to_school :
  ∃ (D : ℝ) (T : ℝ), 
    3 * (T + 7 / 60) = D ∧
    6 * (T - 8 / 60) = D ∧
    D = 1.5 :=
by
  sorry

end distance_home_to_school_l411_411189


namespace prove_f_10_l411_411746

variable (f : ℝ → ℝ)

-- Conditions from the problem
def condition : Prop := ∀ x : ℝ, f (3 ^ x) = x

-- Statement of the problem
theorem prove_f_10 (h : condition f) : f 10 = Real.log 10 / Real.log 3 :=
by
  sorry

end prove_f_10_l411_411746


namespace inclination_angle_of_line_l411_411498

theorem inclination_angle_of_line :
  ∀ (x y : ℝ), 3 * x + (sqrt 3) * y + 3 = 0 → atan2 y x = 120 * (real.pi / 180) :=
by
  assume x y,
  assume h : 3 * x + (sqrt 3) * y + 3 = 0,
  sorry

end inclination_angle_of_line_l411_411498


namespace number_in_125th_place_with_digit_sum_5_l411_411238

theorem number_in_125th_place_with_digit_sum_5 : 
  ∃ n : ℕ, (∃ (seq : list ℕ), (∀ x ∈ seq, (∀ d : ℕ, d ∈ digits 10 x → 0 ≤ d ∧ d ≤ 9) ∧ (list.sum (digits 10 x) = 5) ∧ (list.sorted (<) seq) ∧ seq.length = 125) ∧ list.nth seq 124 = some 41000) :=
by sorry

end number_in_125th_place_with_digit_sum_5_l411_411238


namespace fraction_subtraction_l411_411299

theorem fraction_subtraction : (5 / 6 + 1 / 4 - 2 / 3) = (5 / 12) := by
  sorry

end fraction_subtraction_l411_411299


namespace right_triangles_count_proof_l411_411703

-- Define the points A, B, C, D, P, Q, and R, and the segments AD and BC, along with the given conditions
variables (A B C D P Q R : Type) 

-- Define the condition for these points being vertices of a rectangle
-- Note: In Lean, we elaborate the structure and relationships rather than simply stating "rectangle"
-- So, we define points and segments accordingly.
variables (rectangle : ∀ (A B C D : Type), Prop)
variables (bisects_AD : ∀ (A D : Type) (P : Type), Prop)
variables (bisects_BC : ∀ (B C : Type) (Q : Type), Prop)
variables (right_triangle_APR : ∀ (A P R : Type), Prop)

-- Given the conditions
axiom condition1 : rectangle A B C D
axiom condition2 : bisects_AD A D P
axiom condition3 : bisects_BC B C Q
axiom condition4 : right_triangle_APR A P R

-- Statement: There are exactly 9 right triangles
noncomputable def count_right_triangles_equals_9 : Prop :=
  ∃ (count : ℕ), count = 9 ∧ 
  ∀ (tr : set (set Type)), 
  (∀ p ∈ tr, ∃ (x y z : Type), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x ∈ tr) ∧ (y ∈ tr) ∧ (z ∈ tr)) →
  (tr.card = 9)

theorem right_triangles_count_proof : count_right_triangles_equals_9 A B C D P Q R rectangle bisects_AD bisects_BC right_triangle_APR := 
sorry

end right_triangles_count_proof_l411_411703


namespace number_2007_position_l411_411243

/--
Arrange the positive odd numbers in 5 columns as described in the problem statement.
Given this arrangement pattern of numbers, the proof aims to show that the number 2007
is located in Row 251, Column 5.
-/
theorem number_2007_position :
  ∃ row col, odd_num_in_position 2007 row col ∧ row = 251 ∧ col = 5 :=
sorry

/--
Defines the predicate for the placement of numbers given the odd number and its row and column
such that the arrangement follows the two rows eight numbers cycle pattern.
-/
def odd_num_in_position (num row col : ℕ) : Prop :=
  ∃ k, num = 1 + 2 * k ∧ 
  ((row = 2 * k / 8 + 1 ∧ col = (2 * k % 8) / 2 + 1) ∨
   (row = 2 * k / 8 + 2 ∧ col = 5 - (2 * k % 8) / 2))


end number_2007_position_l411_411243


namespace eight_percent_of_fifty_is_four_l411_411659

theorem eight_percent_of_fifty_is_four : 0.08 * 50 = 4 := by
  sorry

end eight_percent_of_fifty_is_four_l411_411659


namespace bread_slices_leftover_l411_411092

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end bread_slices_leftover_l411_411092


namespace cone_lateral_surface_area_l411_411384

theorem cone_lateral_surface_area (r V: ℝ) (h : ℝ) (l : ℝ) (L: ℝ):
  r = 3 →
  V = 12 * Real.pi →
  V = (1 / 3) * Real.pi * r^2 * h →
  l = Real.sqrt (r^2 + h^2) →
  L = Real.pi * r * l →
  L = 15 * Real.pi :=
by
  intros hr hv hV hl hL
  rw [hr, hv] at hV
  sorry

end cone_lateral_surface_area_l411_411384


namespace sum_factorials_last_two_digits_l411_411973

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l411_411973


namespace negation_of_existence_l411_411085

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 > 2) ↔ ∀ x : ℝ, x^2 ≤ 2 :=
by
  sorry

end negation_of_existence_l411_411085


namespace increasing_interval_of_function_l411_411500

noncomputable def interval_increasing (y : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {x | a ≤ x ∧ x ≤ b ∧ deriv y x > 0}

theorem increasing_interval_of_function:
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ real.pi) →
    (interval_increasing (λ x, 2 * real.sin (real.pi / 6 - 2 * x)) (real.pi / 3) (5 * real.pi / 6)) x :=
by
  sorry

end increasing_interval_of_function_l411_411500


namespace sequence_property_l411_411446

theorem sequence_property {m : ℤ} (h_m : |m| ≥ 2) (a : ℕ → ℤ)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) - m * a n)
  (r s : ℕ) (h_r_s : r > s ∧ s ≥ 2) (h_eq : a r = a s ∧ a s = a 1) :
  r - s ≥ |m| := sorry

end sequence_property_l411_411446


namespace swimming_pool_distance_l411_411419

theorem swimming_pool_distance (julien_daily_distance : ℕ) (sarah_multi_factor : ℕ)
    (jamir_additional_distance : ℕ) (week_days : ℕ) 
    (julien_weekly_distance : ℕ) (sarah_weekly_distance : ℕ) (jamir_weekly_distance : ℕ) 
    (total_combined_distance : ℕ) : 
    julien_daily_distance = 50 → 
    sarah_multi_factor = 2 →
    jamir_additional_distance = 20 →
    week_days = 7 →
    julien_weekly_distance = julien_daily_distance * week_days →
    sarah_weekly_distance = (sarah_multi_factor * julien_daily_distance) * week_days →
    jamir_weekly_distance = ((sarah_multi_factor * julien_daily_distance) + jamir_additional_distance) * week_days →
    total_combined_distance = julien_weekly_distance + sarah_weekly_distance + jamir_weekly_distance →
    total_combined_distance = 1890 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4] at *
  rw [h5, h6, h7, h8]
  sorry

end swimming_pool_distance_l411_411419


namespace fg_minus_gf_l411_411911

noncomputable def f (x : ℝ) : ℝ := 8 * x - 12
noncomputable def g (x : ℝ) : ℝ := x / 4 - 1

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -16 :=
by
  -- We skip the proof.
  sorry

end fg_minus_gf_l411_411911


namespace balance_balls_l411_411873

noncomputable def green_weight := (9 : ℝ) / 4
noncomputable def yellow_weight := (7 : ℝ) / 3
noncomputable def white_weight := (3 : ℝ) / 2

theorem balance_balls (B : ℝ) : 
  5 * green_weight * B + 4 * yellow_weight * B + 3 * white_weight * B = (301 / 12) * B :=
by
  sorry

end balance_balls_l411_411873


namespace problem_ACD_l411_411310

theorem problem_ACD (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2 ∧ π/2 < β ∧ β < π)
  (h2 : sin α = (√5)/5)
  (h3 : cos (α + β) = - (2 * √5)/5) : 
  cos α = (2 * √5)/5 ∧ 
  cos β = -3/5 ∧ 
  sin (α - β) = -(11 * √5) / 25 := 
by
  sorry

end problem_ACD_l411_411310


namespace total_amount_received_l411_411998

theorem total_amount_received (P R CI: ℝ) (T: ℕ) 
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (P_eq: P = 2828.80 / 0.1664) 
  (R_eq: R = 8) 
  (T_eq: T = 2) : 
  P + CI = 19828.80 := 
by 
  sorry

end total_amount_received_l411_411998


namespace sin_13pi_over_6_equals_half_l411_411279

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l411_411279


namespace percentage_change_of_area_l411_411999

theorem percentage_change_of_area (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
    let Area_original := L * B in
    let Length_new := L / 2 in
    let Breadth_new := 3 * B in
    let Area_new := Length_new * Breadth_new in
    (Area_new - Area_original) / Area_original * 100 = 50 :=
by
    sorry

end percentage_change_of_area_l411_411999


namespace find_a_l411_411756

theorem find_a (a : ℝ) (h1 : a^2 + 2 * a - 15 = 0) (h2 : a^2 + 4 * a - 5 ≠ 0) :
  a = 3 :=
by
sorry

end find_a_l411_411756


namespace ratio_of_doctors_lawyers_engineers_l411_411487

variables (d l e : ℕ)

-- Conditions
def average_age_per_group (d l e : ℕ) : Prop :=
  (40 * d + 55 * l + 35 * e) = 45 * (d + l + e)

-- Theorem
theorem ratio_of_doctors_lawyers_engineers
  (h : average_age_per_group d l e) :
  l = d + 2 * e :=
by sorry

end ratio_of_doctors_lawyers_engineers_l411_411487


namespace sum_of_largest_and_smallest_l411_411954

theorem sum_of_largest_and_smallest (d1 d2 d3 d4 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 3) (h4 : d4 = 9) :
  let largest := 9631
  let smallest := 1369
  largest + smallest = 11000 :=
by
  let largest := 9631
  let smallest := 1369
  sorry

end sum_of_largest_and_smallest_l411_411954


namespace last_two_digits_sum_factorials_l411_411965

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l411_411965


namespace solve_equation_1_solve_equation_2_l411_411100

theorem solve_equation_1 (x : ℝ) :
  x^2 - 10 * x + 16 = 0 → x = 8 ∨ x = 2 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 3) = 6 - 2 * x → x = 3 ∨ x = -2 :=
by
  sorry

end solve_equation_1_solve_equation_2_l411_411100


namespace four_unit_vectors_sum_zero_to_opposite_pairs_l411_411520

variables {V : Type} [AddCommGroup V] [Module ℝ V]

theorem four_unit_vectors_sum_zero_to_opposite_pairs
  (u1 u2 u3 u4 : V)
  (h1 : ∥u1∥ = 1)
  (h2 : ∥u2∥ = 1)
  (h3 : ∥u3∥ = 1)
  (h4 : ∥u4∥ = 1)
  (h_sum : u1 + u2 + u3 + u4 = 0) :
  ∃ (v1 v2 v3 v4 : V), (v1 + v3 = 0) ∧ (v2 + v4 = 0) := sorry

end four_unit_vectors_sum_zero_to_opposite_pairs_l411_411520


namespace no_nonneg_int_exists_6n_plus_19_prime_l411_411268

theorem no_nonneg_int_exists_6n_plus_19_prime :
  ¬ ∃ n : ℕ, prime (6^n + 19) :=
sorry

end no_nonneg_int_exists_6n_plus_19_prime_l411_411268


namespace line_intercepts_and_slope_l411_411499

theorem line_intercepts_and_slope :
  ∀ (x y : ℝ), (4 * x - 5 * y - 20 = 0) → 
  ∃ (x_intercept : ℝ) (y_intercept : ℝ) (slope : ℝ), 
    x_intercept = 5 ∧ y_intercept = -4 ∧ slope = 4 / 5 :=
by
  sorry

end line_intercepts_and_slope_l411_411499


namespace integer_elements_in_C_union_A_l411_411354

open Set

def U : Set ℝ := Set.univ
def Z : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = ↑n}
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}
variable (C : Set ℝ)

theorem integer_elements_in_C_union_A :
  ∀ U = set.univ, (Z ∩ (C ∪ A)).to_finset.card = 4 := 
sorry

end integer_elements_in_C_union_A_l411_411354


namespace simplify_rationalize_denominator_l411_411905

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411905


namespace solve_equation_l411_411175

theorem solve_equation : ∃ x : ℚ, x + 5 / 8 = 1 / 4 - 2 / 5 + 7 / 10 ∧ x = -3 / 40 :=
by
  use -3 / 40
  split
  · linarith
  · rfl

end solve_equation_l411_411175


namespace adjacent_sum_correct_l411_411132

noncomputable def sum_of_adjacent_to_seven_in_circle_with_common_factors : ℕ :=
  147 -- The correct sum according to the problem's conditions and solution.

theorem adjacent_sum_correct :
  (∀ (x y : ℕ), x ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧ y ∈ {2, 4, 7, 14, 28, 49, 98, 196} →
    x ≠ y → nat.gcd x y > 1) →
  ∃ (a b : ℕ), a ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧ b ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧
    a ≠ 7 ∧ b ≠ 7 ∧ nat.gcd a 7 > 1 ∧ nat.gcd b 7 > 1 ∧ a + b = sum_of_adjacent_to_seven_in_circle_with_common_factors :=
begin
  sorry
end

end adjacent_sum_correct_l411_411132


namespace original_area_is_2sqrt6_l411_411801

-- Given conditions
def side_length : ℝ := 2
def projection_area : ℝ := (sqrt 3) / 4 * side_length^2

-- Relationship between area of original figure and projection
def original_area (S : ℝ) : ℝ := 2 * sqrt 2 * S

-- Proof statement
theorem original_area_is_2sqrt6 : original_area projection_area = 2 * sqrt 6 := by
  sorry

end original_area_is_2sqrt6_l411_411801


namespace derivative_of_y_eqn_l411_411116

noncomputable def y (x : ℝ) : ℝ := (sin x) - (2 ^ x)

theorem derivative_of_y_eqn (x : ℝ) :
  (deriv y x) = (cos x - (2 ^ x) * log 2) :=
by 
  -- Proof skipped
  sorry

end derivative_of_y_eqn_l411_411116


namespace number_of_gate_tickets_l411_411609

theorem number_of_gate_tickets 
  (pre_bought_tickets : ℕ) (pre_bought_price : ℕ) (gate_price : ℕ) (extra_paid : ℕ) 
  (total_pre_bought_paid : pre_bought_tickets * pre_bought_price)
  (total_gate_paid : ℕ) :
  total_gate_paid = total_pre_bought_paid + extra_paid →
  total_gate_paid = 200 * 30 :=
by
  intros h
  sorry

end number_of_gate_tickets_l411_411609


namespace minimum_area_of_triangle_l411_411410

noncomputable def minimum_area_triangle : ℕ :=
  5.

theorem minimum_area_of_triangle (p q : ℤ) :
  let A : ℤ × ℤ := (0, 0)
  let B : ℤ × ℤ := (30, 10)
  let C : ℤ × ℤ := (p, q)
  -- formula for the area of the triangle using Shoelace theorem
  (∃ (p q : ℤ), 2 * abs (30 * q - 10 * p) = 10) →
  minimum_area_triangle = 5 :=
by sorry

end minimum_area_of_triangle_l411_411410


namespace compare_t1_t2_l411_411472

variable {d m n : ℝ} (h_mn : m ≠ n)

def t1 (d m n : ℝ) : ℝ := 2 * (d / (m + n))

def t2 (d m n : ℝ) : ℝ := (d / (2 * m)) + (d / (2 * n))

theorem compare_t1_t2 (h_mn : m ≠ n) : t1 d m n < t2 d m n :=
by
  sorry

end compare_t1_t2_l411_411472


namespace prob_two_ones_in_twelve_dice_l411_411566

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411566


namespace polar_to_cartesian_and_angle_l411_411351

theorem polar_to_cartesian_and_angle (t : ℝ) (α : ℝ) :
  (∀ θ ρ, ρ = 4 * real.cos θ →
    (let ⟨x, y⟩ := (ρ * real.cos θ, ρ * real.sin θ) in
     (x - 2)^2 + y^2 = 4)) ∧
  ((1 + t * real.cos α - 2)^2 + (t * real.sin α)^2 = 4 →
    |2 * real.cos α| = sqrt 2 → 
    (α = real.pi / 4 ∨ α = 3 * real.pi / 4)) :=
by sorry

end polar_to_cartesian_and_angle_l411_411351


namespace swimming_pool_distance_l411_411421

theorem swimming_pool_distance (julien_daily_distance : ℕ) (sarah_multi_factor : ℕ)
    (jamir_additional_distance : ℕ) (week_days : ℕ) 
    (julien_weekly_distance : ℕ) (sarah_weekly_distance : ℕ) (jamir_weekly_distance : ℕ) 
    (total_combined_distance : ℕ) : 
    julien_daily_distance = 50 → 
    sarah_multi_factor = 2 →
    jamir_additional_distance = 20 →
    week_days = 7 →
    julien_weekly_distance = julien_daily_distance * week_days →
    sarah_weekly_distance = (sarah_multi_factor * julien_daily_distance) * week_days →
    jamir_weekly_distance = ((sarah_multi_factor * julien_daily_distance) + jamir_additional_distance) * week_days →
    total_combined_distance = julien_weekly_distance + sarah_weekly_distance + jamir_weekly_distance →
    total_combined_distance = 1890 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4] at *
  rw [h5, h6, h7, h8]
  sorry

end swimming_pool_distance_l411_411421


namespace parallelogram_sides_l411_411512

theorem parallelogram_sides (x y : ℝ) 
    (h1 : 4 * y + 2 = 12) 
    (h2 : 6 * x - 2 = 10)
    (h3 : 10 + 12 + (6 * x - 2) + (4 * y + 2) = 68) :
    x + y = 4.5 := 
by
  -- Proof to be provided
  sorry

end parallelogram_sides_l411_411512


namespace find_m_l411_411798

def setA (m : ℝ) : Set ℝ := {1, m - 2}
def setB : Set ℝ := {2}

theorem find_m (m : ℝ) (H : setA m ∩ setB = {2}) : m = 4 :=
by
  sorry

end find_m_l411_411798


namespace probability_two_ones_in_twelve_dice_l411_411552

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411552


namespace yogurt_cost_l411_411155

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l411_411155


namespace rationalize_denominator_l411_411896

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411896


namespace find_shorter_parallel_side_l411_411720

variable (x : ℝ) (a : ℝ) (b : ℝ) (h : ℝ)

def is_trapezium_area (a b h : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * (a + b) * h

theorem find_shorter_parallel_side
  (h28 : a = 28)
  (h15 : h = 15)
  (hArea : area = 345)
  (hIsTrapezium : is_trapezium_area a b h area):
  b = 18 := 
sorry

end find_shorter_parallel_side_l411_411720


namespace find_operation_l411_411141

theorem find_operation (a b : ℝ) (h_a : a = 0.137) (h_b : b = 0.098) :
  ((a + b) ^ 2 - (a - b) ^ 2) / (a * b) = 4 :=
by
  sorry

end find_operation_l411_411141


namespace maximum_value_l411_411056

open Complex

theorem maximum_value (γ δ : ℂ) (h1 : Complex.abs δ = 1) (h2 : δ * conj γ ≠ -1) :
  Complex.abs ((δ + γ) / (1 + conj γ * δ)) ≤ 1 :=
sorry

end maximum_value_l411_411056


namespace monotonic_increasing_interval_of_f_l411_411505

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logb (1/2) (x^2))

theorem monotonic_increasing_interval_of_f : 
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 0 ∧ -1 ≤ x₂ ∧ x₂ < 0 ∧ x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧ 
  (∀ x : ℝ, f x ≥ 0) := sorry

end monotonic_increasing_interval_of_f_l411_411505


namespace michael_should_remove_eight_scoops_l411_411867

def total_flour : ℝ := 8
def required_flour : ℝ := 6
def scoop_size : ℝ := 1 / 4

theorem michael_should_remove_eight_scoops :
  (total_flour - required_flour) / scoop_size = 8 :=
by
  sorry

end michael_should_remove_eight_scoops_l411_411867


namespace finite_negatives_condition_l411_411796

-- Define the sequence terms
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + n * d

-- Define the condition for finite negative terms
def has_finite_negatives (a1 d : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, arithmetic_seq a1 d n ≥ 0

-- Theorem that proves the desired statement
theorem finite_negatives_condition (a1 d : ℝ) (h1 : a1 < 0) (h2 : d > 0) :
  has_finite_negatives a1 d :=
sorry

end finite_negatives_condition_l411_411796


namespace no_infinite_sequence_of_lines_l411_411466

theorem no_infinite_sequence_of_lines 
  (l : ℕ → (ℝ × ℝ) → ℝ → Prop)
  (k : ℕ → ℝ)
  (a b : ℕ → ℝ) :
  (∀ n, l n (1, 1) k n) →
  (∀ n, k (n + 1) = a n - b n) →
  (∀ n, k n * k (n + 1) ≥ 0) →
  ¬(∃ (seq : ℕ → ℝ),
    (∀ n : ℕ, l n (1, 1) (seq n))
    ∧ (∀ n : ℕ, seq (n + 1) = a n - b n)
    ∧ (∀ n : ℕ, seq n * seq (n + 1) ≥ 0)) :=
sorry

end no_infinite_sequence_of_lines_l411_411466


namespace last_two_digits_sum_factorials_l411_411968

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l411_411968


namespace annual_decrease_l411_411714

theorem annual_decrease (price_2001 price_2009 : ℕ) (years : ℕ) : 
  price_2001 = 1950 → 
  price_2009 = 1670 → 
  years = 8 → 
  (price_2001 - price_2009) / years = 35 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end annual_decrease_l411_411714


namespace max_x_plus_reciprocal_l411_411940

theorem max_x_plus_reciprocal (x : ℝ) (S : Finset ℝ) (hS : S.card = 2023) (h_positive : ∀ y ∈ S, 0 < y)
  (h_sum : ∑ y in S, y = 2024) (h_reciprocal_sum : ∑ y in S, 1 / y = 2024) (hx_in_S : x ∈ S) :
  x + 1 / x ≤ 4048025 / 2024 :=
sorry

end max_x_plus_reciprocal_l411_411940


namespace simplify_rationalize_denominator_l411_411904

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411904


namespace fraction_increase_by_3_l411_411797

theorem fraction_increase_by_3 (x y : ℝ) (h₁ : x' = 3 * x) (h₂ : y' = 3 * y) : 
  (x' * y') / (x' - y') = 3 * (x * y) / (x - y) :=
by
  sorry

end fraction_increase_by_3_l411_411797


namespace tangents_with_slope_one_l411_411241

noncomputable def find_tangents_with_slope_one : ℤ :=
  let f := λ x : ℝ, x^3
  let deriv := λ x : ℝ, 3 * x^2
  let tangent_slope := 1
  let x1 := real.sqrt(1 / 3)
  let x2 := -real.sqrt(1 / 3)
  let valid_x := x1 ≠ x2
  in if valid_x then 2 else 0

theorem tangents_with_slope_one :
  ∃ (n : ℤ), n = 2 ∧ find_tangents_with_slope_one = n :=
by
  use 2
  sorry

end tangents_with_slope_one_l411_411241


namespace net_profit_is_correct_l411_411137

-- Define the purchase price, markup, and overhead percentage
def purchase_price : ℝ := 48
def markup : ℝ := 55
def overhead_percentage : ℝ := 0.30

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the net profit calculation
def net_profit : ℝ := markup - overhead_cost

-- State the theorem
theorem net_profit_is_correct : net_profit = 40.60 :=
by
  sorry

end net_profit_is_correct_l411_411137


namespace count_valid_bases_l411_411304

def is_valid_base (b : ℕ) : Prop :=
  2 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0

theorem count_valid_bases :
  (Finset.range 9).filter (λ b, is_valid_base (b + 2)).card = 6 :=
by
  sorry

end count_valid_bases_l411_411304


namespace probability_two_dice_showing_1_l411_411601

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411601


namespace cube_inscribed_sphere_surface_area_l411_411204

theorem cube_inscribed_sphere_surface_area (V : ℝ) (hV : V = 8) : 
  let a := (V)^(1/3)
  let R := (real.sqrt 3 * a) / 2
  4 * real.pi * R^2 = 12 * real.pi := 
by
  -- Given the volume of the cube is 8
  have ha : a = 2 := by sorry
  -- Calculate the radius of the sphere
  have hR : R = real.sqrt 3 := by sorry
  -- Calculate the surface area of the sphere
  have hS : 4 * real.pi * R^2 = 12 * real.pi := by sorry
  exact hS

end cube_inscribed_sphere_surface_area_l411_411204


namespace coloring_parity_determining_l411_411228

def equilateral_triangle (n : ℕ) := set (fin n × fin n)

def is_adjacent {n : ℕ} (c1 c2 : fin n × fin n) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 = c2.2 - 1)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 = c2.1 - 1)) ∨
  ((c1.1 + c1.2 = c2.1 + c2.2) ∧ (c1.1 = c2.1 + 1 ∨ c1.1 = c2.1 - 1))

def nice_cells {n : ℕ} (T : equilateral_triangle n) (cell : fin n × fin n) : Prop :=
  finset.card (finset.filter (λ c, is_adjacent cell c) T) % 2 = 1

def neutral_cells {n : ℕ} (T : equilateral_triangle n) (cell : fin n × fin n) : Prop :=
  finset.card (finset.filter (λ c, is_adjacent cell c) T) % 2 = 0

def coloring_parity {n : ℕ} (S : finset (fin n × fin n)) :=
  ∀ coloring : finset (fin n × fin n), 
  finset.card (S ∩ coloring) % 2 = 
  (finset.card (S) % 2 + finset.card ((@equilateral_triangle n) \ S) % 2) % 2

theorem coloring_parity_determining (n : ℕ) (S : finset (fin (n + 1) × fin (n + 1)))  :
  n = 2022 → 
  finset.card S = 12120 →
  ∃ T : equilateral_triangle (n + 1), 
    (S ⊆ T) ∧ coloring_parity S :=
by
  intros h₁ h₂
  sorry

end coloring_parity_determining_l411_411228


namespace x_minus_y_values_l411_411743

theorem x_minus_y_values (x y : ℝ) (h₁ : |x + 1| = 4) (h₂ : (y + 2)^2 = 4) (h₃ : x + y ≥ -5) :
  x - y = -5 ∨ x - y = 3 ∨ x - y = 7 :=
by
  sorry

end x_minus_y_values_l411_411743


namespace tangent_and_normal_lines_l411_411285

noncomputable def tangent_line_eq (a : ℝ) : ℝ → ℝ := 
  λ x, - (2 * x) / π + (a * π) / 2

noncomputable def normal_line_eq (a : ℝ) : ℝ → ℝ := 
  λ x, (π * x) / 2 + (a * π) / 2

theorem tangent_and_normal_lines (a : ℝ) : 
  (∃ (x y : ℝ), 
    x = a * (π / 2) * cos (π / 2) ∧ y = a * (π / 2) * sin (π / 2)) →
  (∃ (x y : ℝ), 
    y = tangent_line_eq a x ∨ y = normal_line_eq a x) :=
by
  sorry

end tangent_and_normal_lines_l411_411285


namespace adjacent_integer_sum_l411_411134

theorem adjacent_integer_sum (d195 : Finset ℕ := {2, 4, 7, 14, 28, 49, 98, 196}) :
  (∃ a b ∈ d195, a ≠ b ∧ gcd a 7 > 1 ∧ gcd b 7 > 1) → a + b = 63 := 
by {
    sorry
}

end adjacent_integer_sum_l411_411134


namespace smallest_interval_for_joint_probability_l411_411508

variable {Ω : Type} [MeasureSpace Ω]

def probability (A B : Set Ω) (μ : Measure Ω) : Prop :=
  μ A = 5/6 ∧ μ B = 3/4 ∧ (0 ≤ μ (A ∩ B) ∧ μ (A ∩ B) ≤ 3/4)

theorem smallest_interval_for_joint_probability (A B : Set Ω) (μ : Measure Ω) :
  μ A = 5/6 ∧ μ B = 3/4 → (0 ≤ μ (A ∩ B) ∧ μ (A ∩ B) ≤ 3/4) :=
by
  intro h
  sorry

end smallest_interval_for_joint_probability_l411_411508


namespace solution_part1_solution_part2_l411_411198

-- Part 1
noncomputable def find_a_part1 (P : Set ℝ) (Q : Set ℝ) (f : ℝ → ℝ) (x : ℝ) := 
  P = { x | (1/2 : ℝ) ≤ x ∧ x ≤ 3 } ∧ 
  f = (λ x, Real.log 2 (a * x^2 - 2 * x + 2)) ∧ 
  Q = { x | -2 < x ∧ x < 2/3 } ∧ 
  (P ∩ Q = { x | (1/2 : ℝ) ≤ x ∧ x < 2/3 }) ∧ 
  (P ∪ Q = { x | -2 < x ∧ x ≤ 3 })

theorem solution_part1 (P Q : Set ℝ) (f : ℝ → ℝ) (x : ℝ) : 
  find_a_part1 P Q f x → a = -3/2 :=
by { sorry }

-- Part 2
noncomputable def find_a_part2 (f : ℝ → ℝ) (x : ℝ) := 
  (∀ x, f(x + 3) = f(x)) ∧ 
  (∀ x, 1/2 ≤ x ∧ x ≤ 3 → f(x) = Real.log 2 (a * x^2 - 2 * x + 2)) ∧ 
  f(35) = 1

theorem solution_part2 (f : ℝ → ℝ) (x : ℝ) : 
  find_a_part2 f x → a = 1 :=
by { sorry }

end solution_part1_solution_part2_l411_411198


namespace shifted_polynomial_sum_l411_411986

theorem shifted_polynomial_sum :
  let f (x : ℝ) := 3 * x^2 - 2 * x + 5
  let g (x : ℝ) := f (x + 6)
  let a : ℝ := 3
  let b : ℝ := 34
  let c : ℝ := 101
  in a + b + c = 138 :=
by
  sorry

end shifted_polynomial_sum_l411_411986


namespace probability_exactly_two_ones_l411_411562

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411562


namespace probability_exactly_two_ones_l411_411563

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411563


namespace natural_number_base_conversion_l411_411657

theorem natural_number_base_conversion (n : ℕ) (h7 : n = 4 * 7 + 1) (h9 : n = 3 * 9 + 2) : 
  n = 3 * 8 + 5 := 
by 
  sorry

end natural_number_base_conversion_l411_411657


namespace probability_exactly_two_ones_l411_411561

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411561


namespace intersection_is_0_3_l411_411754

def setA : Set ℝ := {x | x - 1 < 2}
def setB : Set ℝ := {x | 1 < 2^x ∧ 2^x < 16}

theorem intersection_is_0_3 : setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_is_0_3_l411_411754


namespace arithmetic_sequence_sum_first_five_terms_l411_411401

theorem arithmetic_sequence_sum_first_five_terms:
  ∀ (a : ℕ → ℤ), a 2 = 1 → a 4 = 7 → (a 1 + a 5 = a 2 + a 4) → (5 * (a 1 + a 5) / 2 = 20) :=
by
  intros a h1 h2 h3
  sorry

end arithmetic_sequence_sum_first_five_terms_l411_411401


namespace ratio_of_perimeters_l411_411482

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l411_411482


namespace number_in_125th_place_with_digit_sum_5_l411_411237

theorem number_in_125th_place_with_digit_sum_5 : 
  ∃ n : ℕ, (∃ (seq : list ℕ), (∀ x ∈ seq, (∀ d : ℕ, d ∈ digits 10 x → 0 ≤ d ∧ d ≤ 9) ∧ (list.sum (digits 10 x) = 5) ∧ (list.sorted (<) seq) ∧ seq.length = 125) ∧ list.nth seq 124 = some 41000) :=
by sorry

end number_in_125th_place_with_digit_sum_5_l411_411237


namespace least_number_to_subtract_l411_411287

theorem least_number_to_subtract (n : ℕ) (h : n = 964807) : ∃ m : ℕ, n % 8 = m ∧ n - m ≡ 0 [MOD 8] :=
by
  use 7
  rw h
  have : 964807 % 8 = 7 := by norm_num
  split
  · exact this
  · rw [← sub_eq_add_neg, add_comm, Nat.sub_add_cancel]
    sorry

end least_number_to_subtract_l411_411287


namespace axis_of_symmetry_l411_411118

def function : ℝ → ℝ := λ x, sin(4 * x - π / 3)

theorem axis_of_symmetry :
  ∃ x : ℝ, function (x + π / (2 * 4)) = function x :=
begin
  use 11 * π / 24,
  sorry
end

end axis_of_symmetry_l411_411118


namespace probability_two_ones_in_twelve_dice_l411_411585
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411585


namespace sum_of_two_digit_odd_numbers_l411_411516

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end sum_of_two_digit_odd_numbers_l411_411516


namespace popularity_order_l411_411515

def event := Type
def Kickball : event := sorry
def Picnic : event := sorry
def Softball : event := sorry
  
def fraction_liking (e : event) : ℚ :=
  if e = Kickball then 11 / 30
  else if e = Picnic then 7 / 20
  else if e = Softball then 5 / 12
  else 0

theorem popularity_order (most least : list event) :
  most = [Softball, Kickball, Picnic] :=
by {
  sorry
}

end popularity_order_l411_411515


namespace solve_sqrt_equation_l411_411099

theorem solve_sqrt_equation : {x : ℝ | sqrt (2 * x - 4) - sqrt (x + 5) = 1} = {4, 20} :=
by
  sorry

end solve_sqrt_equation_l411_411099


namespace find_S30_l411_411521

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
axiom S_def (n : ℕ) : S n = ∑ i in Finset.range (n+1), a i
axiom S10 : S 10 = 31
axiom S20 : S 20 = 122

-- The theorem (the proof problem)
theorem find_S30 : S 30 = 273 :=
by
  sorry

end find_S30_l411_411521


namespace prob_two_ones_in_twelve_dice_l411_411571

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411571


namespace highest_possible_extensions_in_use_l411_411495

theorem highest_possible_extensions_in_use : 
    let total_extensions := 100
        same_digit_extensions := 10
        distinct_digit_pairs := 90
        usable_distinct_pairs := distinct_digit_pairs / 2
    in same_digit_extensions + usable_distinct_pairs = 55 :=
by
    let total_extensions := 100
    let same_digit_extensions := 10
    let distinct_digit_pairs := 90
    let usable_distinct_pairs := distinct_digit_pairs / 2
    show same_digit_extensions + usable_distinct_pairs = 55
    sorry

end highest_possible_extensions_in_use_l411_411495


namespace find_hyperbola_equation_l411_411041

open Real

variable (F1 F2 : Point ℝ) 
variable (P : Point ℝ)

def isHyperbola (F1 F2 P : Point ℝ) : Prop :=
  let PF1 := dist P F1
  let PF2 := dist P F2
  PF1 * PF2 = 2 ∧ (P - F1) • (P - F2) = 0

theorem find_hyperbola_equation (F1 F2 : Point ℝ) (hF1 : F1 = ⟨-sqrt 10, 0⟩) (hF2 : F2 = ⟨sqrt 10, 0⟩)
  (P : Point ℝ) (h : isHyperbola F1 F2 P) :
  ∃ a b : ℝ, (a = 3 ∧ b = 1 ∧ (P.x ^ 2) / a ^ 2 - (P.y ^ 2) / b ^ 2 = 1) :=
by
  sorry

end find_hyperbola_equation_l411_411041


namespace triangle_area_l411_411673

def point := (ℚ × ℚ)

def vertex1 : point := (3, -3)
def vertex2 : point := (3, 4)
def vertex3 : point := (8, -3)

theorem triangle_area :
  let base := (vertex3.1 - vertex1.1 : ℚ)
  let height := (vertex2.2 - vertex1.2 : ℚ)
  (base * height / 2) = 17.5 :=
by
  sorry

end triangle_area_l411_411673


namespace red_cards_needed_l411_411016

-- Define the initial conditions
def total_players : ℕ := 11
def players_without_cautions : ℕ := 5
def yellow_cards_per_player : ℕ := 1
def yellow_cards_per_red_card : ℕ := 2

-- Theorem statement for the problem
theorem red_cards_needed (total_players = 11) (players_without_cautions = 5) 
    (yellow_cards_per_player = 1) (yellow_cards_per_red_card = 2) : (total_players - players_without_cautions) * yellow_cards_per_player / yellow_cards_per_red_card = 3 := 
by
  sorry

end red_cards_needed_l411_411016


namespace accurate_estimate_of_population_distribution_l411_411688

-- Define the main theorem to prove the correct statement ③
theorem accurate_estimate_of_population_distribution
  (sample_size : ℕ) 
  (population_distribution : ℝ → ℝ) 
  (sample_distribution : ℕ → ℝ → ℝ)
  (larger_sample_size accuracy) :
  (larger_sample_size > sample_size) →
  (sample_distribution larger_sample_size ≈ population_distribution) →
  (sample_distribution sample_size ≈ population_distribution) →
  accuracy larger_sample_size > accuracy sample_size :=
sorry

end accurate_estimate_of_population_distribution_l411_411688


namespace problem_solution_l411_411347

def hyperbola (a b : ℝ) : set (ℝ × ℝ) := {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}
def parabola (p : ℝ) : set (ℝ × ℝ) := {q | q.2^2 = 2 * p * q.1}

noncomputable def find_p (a b : ℝ) (e : ℝ) (area : ℝ) : ℝ :=
if h1 : a > 0 ∧ b > 0 ∧ e = 2 ∧ area = sqrt 3 then
  -- here should be the actual calculation but it's omitted
  2
else
  0

theorem problem_solution (a b p : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (hyperbola a b)) (h4 : (parabola p)) (h5 : (sqrt (3) = area)) (h6 : e = 2) :
  find_p a b e area = 2 := sorry

end problem_solution_l411_411347


namespace inscribed_right_triangle_exists_l411_411315

noncomputable theory

-- Definitions of the geometric entities involved.
variables (O : Point) -- center of the circle ω
variables (A B : Point) -- points inside the circle ω
variables (ω : Circle O) -- circle ω centered at O

-- Proposition: There exists a right-angled triangle inscribed in ω with its legs passing through A and B.
theorem inscribed_right_triangle_exists (A B : Point) (ω : Circle Point) :
  ∃ C : Point, on_circle C ω ∧ ∠ A C B = 90 :=
sorry

end inscribed_right_triangle_exists_l411_411315


namespace books_on_shelves_l411_411939

theorem books_on_shelves (total_books upper_books lower_books : ℕ) 
  (h1 : total_books = 180) 
  (h2 : upper_books + lower_books = total_books) 
  (h3 : lower_books + 15 = 2 * (upper_books - 15)) : 
  upper_books = 75 ∧ lower_books = 105 :=
begin
  sorry
end

end books_on_shelves_l411_411939


namespace exactly_one_passes_l411_411613

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end exactly_one_passes_l411_411613


namespace dice_probability_l411_411594

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411594


namespace range_of_a_l411_411767

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1) < a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l411_411767


namespace max_planes_from_15_points_l411_411976

-- Define the problem conditions
def no_four_points_coplanar (P : set (EuclideanSpace ℝ (fin 3))) : Prop :=
  ∀ (a b c d : EuclideanSpace ℝ (fin 3)), {a, b, c, d} ⊆ P → ¬ coplanar {a, b, c, d}

-- Statement of the problem
theorem max_planes_from_15_points (P : set (EuclideanSpace ℝ (fin 3))) (hP : P.card = 15) (h_nfc : no_four_points_coplanar P) : ∃ (n : ℕ), n = 455 := by
  sorry

end max_planes_from_15_points_l411_411976


namespace arithmetic_sequence_common_difference_l411_411402

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, (∀ (a_n : ℕ → ℝ), a_n 1 = 3 ∧ a_n 3 = 7 ∧ (∀ n, a_n n = 3 + (n - 1) * d)) → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l411_411402


namespace count_pairs_leq_6_l411_411792

def positive_nat_pairs (n : ℕ) := 
  {p : ℕ × ℕ // p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ p.1 + p.2 ≤ n ∧ p.1 ≠ p.2}

theorem count_pairs_leq_6 : 
  (positive_nat_pairs 6).to_finset.card = 15 := 
sorry

end count_pairs_leq_6_l411_411792


namespace total_students_in_Lansing_l411_411051

theorem total_students_in_Lansing :
  let num_schools_300 := 20
  let num_schools_350 := 30
  let num_schools_400 := 15
  let students_per_school_300 := 300
  let students_per_school_350 := 350
  let students_per_school_400 := 400
  (num_schools_300 * students_per_school_300 + num_schools_350 * students_per_school_350 + num_schools_400 * students_per_school_400 = 22500) := 
  sorry

end total_students_in_Lansing_l411_411051


namespace rationalize_denominator_l411_411894

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411894


namespace smallest_n_geometric_sequence_l411_411750

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem smallest_n_geometric_sequence :
  ∃ n : ℕ, n > 0 ∧ 
    let a := (λ n, 2^ (n-1)); 
    let b := (λ n, a (n + 1) * Real.log 2 (a (n + 1))) 
    let S := λ n, ∑ i in Finset.range n, b (i + 1)  -- Sum of first n terms
    2^(n+1) + S n > 60 * n + 2 :=
by
  sorry

end smallest_n_geometric_sequence_l411_411750


namespace num_non_perfect_square_sets_l411_411849

def T (i : ℕ) : set ℤ := {n : ℤ | 200 * i ≤ n ∧ n < 200 * (i + 1)}

def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, k * k = n

def is_perfect_square_in_T (i : ℕ) : Prop := ∃ n : ℤ, n ∈ T i ∧ is_perfect_square n

theorem num_non_perfect_square_sets : ∑ i in finset.range 500, if is_perfect_square_in_T i then 0 else 1 = 487 :=
by sorry

end num_non_perfect_square_sets_l411_411849


namespace sum_r_j_eq_3_l411_411509

variable (p r j : ℝ)

theorem sum_r_j_eq_3
  (h : (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21) :
  r + j = 3 := by
  sorry

end sum_r_j_eq_3_l411_411509


namespace symmetric_about_pi_over_4_min_phi_l411_411949

theorem symmetric_about_pi_over_4_min_phi 
  (φ : ℝ) 
  (hφ : φ > 0)
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * sin (2 * x + π / 4))
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = 2 * sin (4 * x - 2 * φ + π / 4)) 
  : (∀ x, g (π / 4 - x) = g (π / 4 + x)) ↔ φ = 3 * π / 8 := 
by
  sorry

end symmetric_about_pi_over_4_min_phi_l411_411949


namespace probability_two_dice_show_1_l411_411573

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411573


namespace dice_probability_l411_411598

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411598


namespace product_of_positive_real_part_solutions_of_x6_eq_neg216_l411_411803

theorem product_of_positive_real_part_solutions_of_x6_eq_neg216:
  let x_solutions := { z : ℂ | z^6 = -216 }
  let positive_real_solutions := { z : ℂ | z ∈ x_solutions ∧ z.re > 0 }
  ∏ z in positive_real_solutions, z = 216i :=
by
  sorry

end product_of_positive_real_part_solutions_of_x6_eq_neg216_l411_411803


namespace alex_jelly_beans_l411_411232

theorem alex_jelly_beans (initial_ounces : ℕ) (eaten_ounces : ℕ)
  (divided_piles : ℕ) (remaining_ounces : ℕ) (pile_weight : ℕ) :
  initial_ounces = 36 →
  eaten_ounces = 6 →
  divided_piles = 3 →
  remaining_ounces = initial_ounces - eaten_ounces →
  pile_weight = remaining_ounces / divided_piles →
  pile_weight = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  have h6 : remaining_ounces = 30 := by linarith [h4]
  rw [h3, h6] at h5
  have h7 : pile_weight = 10 := by linarith [h5]
  exact h7

end alex_jelly_beans_l411_411232


namespace find_q_l411_411779

variable (p q : ℝ) (hp : p > 1) (hq : q > 1) (h_cond1 : 1 / p + 1 / q = 1) (h_cond2 : p * q = 9)

theorem find_q : q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l411_411779


namespace smallest_number_is_D_l411_411686

-- Definitions for each number in their respective bases
def numA : ℕ := 8 * 9^1 + 5 * 9^0
def numB : ℕ := 2 * 6^2 + 1 * 6^1 + 0 * 6^0
def numC : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0
def numD : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Prove that numD is the smallest among them.
theorem smallest_number_is_D : numD < numA ∧ numD < numB ∧ numD < numC :=
by
  -- Asserting the calculated values
  have hA : numA = 77 := by norm_num
  have hB : numB = 78 := by norm_num
  have hC : numC = 64 := by norm_num
  have hD : numD = 63 := by norm_num

  -- Using the calculated values to compare
  rw [hA, hB, hC, hD]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_number_is_D_l411_411686


namespace problem_statement_l411_411433

theorem problem_statement (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6) :
  (a * b / c) + (b * c / a) + (c * a / b) = 49 / 6 := 
by sorry

end problem_statement_l411_411433


namespace perimeter_of_triangle_l411_411494

def is_ellipse (a : ℝ) : Prop :=
  (a > 3) ∧ (∃ x y, (x^2 / a^2) + (y^2 / 9) = 1)

def distance_between_foci : ℝ := 8

def passes_through (chord AB : ℕ) : Prop :=
  ∃ F1, (AB = F1)

theorem perimeter_of_triangle
  (a : ℝ)
  (h_ellipse : is_ellipse a)
  (h_distance : distance_between_foci = 8)
  (h_chord : passes_through 1) :
  ∃ (perimeter : ℝ), perimeter = 20 :=
begin
  sorry
end

end perimeter_of_triangle_l411_411494


namespace average_people_moving_to_oregon_per_hour_l411_411470

theorem average_people_moving_to_oregon_per_hour :
  let days := 5 in
  let people := 3500 in
  let hours_per_day := 24 in
  let total_hours := days * hours_per_day in
  (people / total_hours).round = 29 := 
by
  sorry

end average_people_moving_to_oregon_per_hour_l411_411470


namespace six_digit_mod7_l411_411815

theorem six_digit_mod7 (a b c d e f : ℕ) (N : ℕ) (h : N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) (h_div7 : N % 7 = 0) :
    (10^5 * f + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e) % 7 = 0 :=
by
  sorry

end six_digit_mod7_l411_411815


namespace no_tiling_6x6_no_straight_cuts_tiling_mn_no_straight_cuts_tiling_6x8_no_straight_cuts_l411_411632

-- Part (a): Prove it is impossible.
theorem no_tiling_6x6_no_straight_cuts :
  ¬ ∃ (tiling : fin 6 × fin 6 → fin 2 × fin 2), 
    (∀ i j, (tiling i j = tiling (i + 1) j) ∨ (tiling i j = tiling i (j + 1))∧
    (∀ line, (∀ i, tiling (i + line % 6) line = tiling (i + 1 + line % 6) line ∨ 
                     tiling line (i + line % 6) = tiling line (i + 1 + line % 6))) :=
sorry

-- Part (b): Prove tiling possibility for any m by n (m, n > 6 and mn even) rectangle.
theorem tiling_mn_no_straight_cuts (m n : ℕ) (hm : 6 < m) (hn : 6 < n) (hmn : even (m * n)) :
  ∃ (tiling : fin m × fin n → fin 2 × fin 2), 
    (∀ x, ∃ a b, tiling x.1 x.2 = (a, b) ∧ ¬ same_line tiling x x) :=
sorry

-- Part (c): Prove tiling possibility for a 6 by 8 rectangle.
theorem tiling_6x8_no_straight_cuts :
  ∃ (tiling : fin 6 × fin 8 → fin 2 × fin 2), 
    (∀ x, ∃ a b, tiling x.1 x.2 = (a, b) ∧ ¬ same_line tiling x x) :=
sorry

end no_tiling_6x6_no_straight_cuts_tiling_mn_no_straight_cuts_tiling_6x8_no_straight_cuts_l411_411632


namespace minimum_workers_required_l411_411259

theorem minimum_workers_required (total_days : ℕ) (days_elapsed : ℕ) (initial_workers : ℕ) (job_fraction_done : ℚ)
  (remaining_work_fraction : job_fraction_done < 1) 
  (worker_productivity_constant : Prop) : 
  total_days = 40 → days_elapsed = 10 → initial_workers = 10 → job_fraction_done = (1/4) →
  (total_days - days_elapsed) * initial_workers * job_fraction_done = (1 - job_fraction_done) →
  job_fraction_done = 1 → initial_workers = 10 :=
by
  intros;
  sorry

end minimum_workers_required_l411_411259


namespace proof_problem_l411_411741

-- Define f(x) as given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := (m * 2^x - 1) / (2^x + 1)

-- Define the property that f(x) is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the condition that f(x-3) + f(a + x^2) > 0 always holds
def condition (a m : ℝ) : Prop := ∀ x : ℝ, f m (x - 3) + f m (a + x^2) > 0

-- The main theorem for Part (1) and Part (2)
theorem proof_problem (m a : ℝ) (h1 : is_odd_function (f m)) (h2 : condition a 1) :
  m = 1 ∧ a > 13 / 4 :=
by 
  sorry

end proof_problem_l411_411741


namespace freshmen_minus_sophomores_eq_24_l411_411996

def total_students := 800
def percent_juniors := 27 / 100
def percent_not_sophomores := 75 / 100
def number_seniors := 160

def number_juniors := percent_juniors * total_students
def number_not_sophomores := percent_not_sophomores * total_students
def number_sophomores := total_students - number_not_sophomores
def number_freshmen := total_students - (number_juniors + number_sophomores + number_seniors)

theorem freshmen_minus_sophomores_eq_24 :
  number_freshmen - number_sophomores = 24 :=
sorry

end freshmen_minus_sophomores_eq_24_l411_411996


namespace card_probability_l411_411234

-- Definitions of the conditions
def is_multiple (n d : ℕ) : Prop := d ∣ n

def count_multiples (d m : ℕ) : ℕ := (m / d)

def multiples_in_range (n : ℕ) : ℕ := 
  count_multiples 2 n + count_multiples 3 n + count_multiples 5 n
  - count_multiples 6 n - count_multiples 10 n - count_multiples 15 n 
  + count_multiples 30 n

def probability_of_multiples_in_range (n : ℕ) : ℚ := 
  multiples_in_range n / n 

-- Proof statement
theorem card_probability (n : ℕ) (h : n = 120) : probability_of_multiples_in_range n = 11 / 15 :=
  sorry

end card_probability_l411_411234


namespace father_son_skating_ratio_l411_411496

theorem father_son_skating_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : (v_f + v_s) / (v_f - v_s) = 5) :
  v_f / v_s = 1.5 :=
sorry

end father_son_skating_ratio_l411_411496


namespace sum_of_roots_l411_411853

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 2 * x1^2 - k * x1 = 2 * c) 
  (h2 : 2 * x2^2 - k * x2 = 2 * c) (h3 : x1 ≠ x2) : x1 + x2 = k / 2 := 
sorry

end sum_of_roots_l411_411853


namespace chi_squared_relationship_l411_411945

theorem chi_squared_relationship (K2_value : ℝ)
    (H_obs : K2_value = 4.844)
    (H_critical_3_841 : P (λ x, x ≥ 3.841) = 0.05)
    (H_critical_5_024 : P (λ x, x ≥ 5.024) = 0.025)
    (H_critical_6_635 : P (λ x, x ≥ 6.635) = 0.01) :
  P (λ x, K2_value ≥ 3.841) = 0.05 :=
by
  -- Proof omitted
  sorry

end chi_squared_relationship_l411_411945


namespace triangle_statements_l411_411833

theorem triangle_statements
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = real.to_radians A)
  (hB : B = real.to_radians B)
  (hC : C = real.to_radians C)
  (ha : real.angle_ABC a b c = A)
  (hb : real.angle_ABC b a c = B)
  (hc : real.angle_ABC c a b = C)
  (htriangle : ℝ.triangle a b c)
  (h_sin_A : sin A)
  (h_sin_B : sin B)
  (h_sin_C : sin C):

  (h1 : (a + b + c) / (h_sin_A + h_sin_B + h_sin_C) = a / h_sin_A) ∧
  (h2 : h_sin_A > h_sin_B → A > B) ∧
  (h3 : A > B → a > b) :=
begin
  sorry
end

end triangle_statements_l411_411833


namespace investment_ratio_l411_411992

theorem investment_ratio (X_investment Y_investment : ℕ) (hX : X_investment = 5000) (hY : Y_investment = 15000) : 
  X_investment * 3 = Y_investment :=
by
  sorry

end investment_ratio_l411_411992


namespace parallel_line_equation_perpendicular_line_equation1_perpendicular_line_equation2_l411_411321

-- Problem Definitions
section LineEquations

-- Given Conditions
def line_l (x y : ℝ) : Prop := x / 4 + y / 3 = 1
def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)
def intersect_point_C : (ℝ × ℝ) := (-1, 3)
def line1 (x : ℝ) : Prop := 3 * x + y = 0
def line2 (x : ℝ) : Prop := x + y = 2

-- Statements to be proved
theorem parallel_line_equation (x y: ℝ) : 
  (line1 x y ∧ line2 x y) → (3 * x + 4 * y - 9 = 0) :=
sorry

theorem perpendicular_line_equation1 (x y: ℝ) : 
  (3 * x + 4 * y - 9 = 0) → (4 * x - 3 * y - 12 = 0) :=
sorry

theorem perpendicular_line_equation2 (x y: ℝ) : 
  (3 * x + 4 * y - 9 = 0) → (4 * x - 3 * y + 12 = 0) :=
sorry

end LineEquations

end parallel_line_equation_perpendicular_line_equation1_perpendicular_line_equation2_l411_411321


namespace harmonic_mean_pairs_220_l411_411733

noncomputable def harmonic_mean (x y : ℕ) : ℚ :=
  2 * x * y / (x + y)

theorem harmonic_mean_pairs_220 :
  -- condition: number of pairs (x, y) such that x < y and harmonic mean is 12^10
  finset.card {p : ℕ × ℕ | p.1 < p.2 ∧ harmonic_mean p.1 p.2 = 12^10} = 220 :=
sorry

end harmonic_mean_pairs_220_l411_411733


namespace prime_abs_a_squared_minus_3a_minus_6_l411_411194

open Int

def abs (x : ℤ) : ℤ := if x < 0 then -x else x

theorem prime_abs_a_squared_minus_3a_minus_6 (a : ℤ) :
  Prime (abs (a^2 - 3 * a - 6)) → (a = -1 ∨ a = 4) := 
sorry

end prime_abs_a_squared_minus_3a_minus_6_l411_411194


namespace ratio_of_tangent_segment_l411_411638

theorem ratio_of_tangent_segment (r : ℝ) (h_pos : 0 < r) :
  let radius_small1 := 2 * r,
      radius_small2 := 3 * r,
      radius_large  := 6 * r,
      diameter_large := 2 * radius_large,
      tangent_length := (12 * r * Real.sqrt 6) / 5 in
  tangent_length / diameter_large = (2 * Real.sqrt 6) / 5 :=
by
  sorry

end ratio_of_tangent_segment_l411_411638


namespace bread_leftover_after_sandwiches_l411_411090

def total_bread_slices (bread_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  bread_packages * slices_per_package

def total_ham_slices (ham_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  ham_packages * slices_per_package

def sandwiches_from_ham (ham_slices: ℕ) : ℕ :=
  ham_slices

def total_bread_used (sandwiches: ℕ) (bread_slices_per_sandwich: ℕ) : ℕ :=
  sandwiches * bread_slices_per_sandwich

def bread_leftover (total_bread: ℕ) (bread_used: ℕ) : ℕ :=
  total_bread - bread_used

theorem bread_leftover_after_sandwiches :
  let bread_packages := 2
  let bread_slices_per_package := 20
  let ham_packages := 2
  let ham_slices_per_package := 8
  let bread_slices_per_sandwich := 2 in
  bread_leftover
    (total_bread_slices bread_packages bread_slices_per_package)
    (total_bread_used
      (sandwiches_from_ham (total_ham_slices ham_packages ham_slices_per_package))
      bread_slices_per_sandwich) = 8 :=
by
  sorry

end bread_leftover_after_sandwiches_l411_411090


namespace exists_n_for_prime_l411_411710

theorem exists_n_for_prime (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ n : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 2 * k + 1 ∧ n = k^2 ∧ Nat.sqrt (p + n) + Nat.sqrt n = Nat.sqrt ((k + 1)^2) + Nat.sqrt (k^2) ∧ Nat.sqrt ((k + 1)^2) + Nat.sqrt (k^2) ∈ ℤ) := 
sorry

end exists_n_for_prime_l411_411710


namespace units_digit_base8_of_sum_34_8_47_8_l411_411728

def is_units_digit (n m : ℕ) (u : ℕ) := (n + m) % 8 = u

theorem units_digit_base8_of_sum_34_8_47_8 :
  ∀ (n m : ℕ), n = 34 ∧ m = 47 → (is_units_digit (n % 8) (m % 8) 3) :=
by
  intros n m h
  rw [h.1, h.2]
  sorry

end units_digit_base8_of_sum_34_8_47_8_l411_411728


namespace mrs_sheridan_gave_away_14_cats_l411_411071

def num_initial_cats : ℝ := 17.0
def num_left_cats : ℝ := 3.0
def num_given_away (x : ℝ) : Prop := num_initial_cats - x = num_left_cats

theorem mrs_sheridan_gave_away_14_cats : num_given_away 14.0 :=
by
  sorry

end mrs_sheridan_gave_away_14_cats_l411_411071


namespace math_team_selection_l411_411458

theorem math_team_selection : 
  (nat.choose 6 3) * (nat.choose 8 3) = 1120 := 
by
  sorry

end math_team_selection_l411_411458


namespace necessary_not_sufficient_condition_l411_411658

variables {Point Line Plane : Type}

-- Definitions for perpendicularity
def perpendicular_line_to_plane (l : Line) (α : Plane) : Prop :=
∀ (m : Line), m ∈ α → perpendicular l m

def perpendicular_plane (β α : Plane) : Prop :=
∀ (l : Line), l ∈ β → perpendicular_line_to_plane l α

-- Condition in problem (some plane passing through line l is perpendicular to plane α)
def condition (l : Line) (α : Plane) : Prop :=
∃ (β : Plane), (l ∈ β) ∧ perpendicular_plane β α

-- Proposition that the condition is a necessary but not sufficient condition for perpendicularity
theorem necessary_not_sufficient_condition {l : Line} {α : Plane} :
  (∃ (β : Plane), l ∈ β ∧ (perpendicular_plane β α)) → ¬ (perpendicular_line_to_plane l α) :=
sorry

end necessary_not_sufficient_condition_l411_411658


namespace chord_length_l411_411202

theorem chord_length
  (r d : ℝ)
  (hr: r = 5)
  (hd: d = 4)
  : ∃ PQ: ℝ, PQ = 2 * real.sqrt (r^2 - d^2) ∧ PQ = 6 :=
  by
  sorry

end chord_length_l411_411202


namespace even_function_periodicity_l411_411332

noncomputable def f : ℝ → ℝ :=
sorry -- The actual function definition is not provided here but assumed to exist.

theorem even_function_periodicity (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2)
  (h2 : f (x + 2) = f x)
  (hf_even : ∀ x, f x = f (-x))
  (hf_segment : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x^2 + 2*x - 1) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2 - 6*x + 7 :=
sorry

end even_function_periodicity_l411_411332


namespace functional_equation_zero_l411_411066

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (hx : ∀ x y : ℝ, f (x + y) = f x + f y) : f 0 = 0 :=
by
  sorry

end functional_equation_zero_l411_411066


namespace sin_13pi_over_6_equals_half_l411_411280

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l411_411280


namespace investment_difference_l411_411695

noncomputable def future_value_simple (principal: ℕ) (rate: ℚ) (time: ℕ) : ℚ :=
  principal * (1 + rate)

noncomputable def future_value_compound (principal: ℕ) (rate: ℚ) (time: ℕ) : ℚ :=
  principal * (1 + rate) ^ time

theorem investment_difference
  (principal_A: ℕ := 2000)
  (principal_B: ℕ := 1000)
  (rate_A: ℚ := 0.12)
  (rate_B: ℚ := 0.30)
  (time: ℕ := 2) :
  let future_value_A := future_value_simple principal_A rate_A time,
      future_value_B := future_value_compound principal_B rate_B time in
  future_value_A - future_value_B = 790 :=
by {
  sorry
}

end investment_difference_l411_411695


namespace angle_between_vectors_l411_411358

variables (a b : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ∥a∥ = sqrt 6
axiom norm_b : ∥b∥ = sqrt 2
axiom dot_condition : (a - b) • b = 1

theorem angle_between_vectors (θ : ℝ) (hθ : cos θ = sqrt 3 / 2) :
  angle a b = 30 :=
by
  sorry

end angle_between_vectors_l411_411358


namespace find_f_l411_411378

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l411_411378


namespace find_a_l411_411766

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

-- Define the derivative of function f with respect to x
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

-- Define the condition for the problem
def condition (a : ℝ) : Prop := f' a 1 = 2

-- The statement to be proved
theorem find_a (a : ℝ) (h : condition a) : a = -3 :=
by {
  -- Proof is omitted
  sorry
}

end find_a_l411_411766


namespace indeterminate_triangle_l411_411389

-- Given conditions
variables {A B C : ℝ}

-- Triangle ABC
def is_triangle_ABC (α β γ : ℝ) : Prop :=
  α + β + γ = π ∧ 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π

-- Condition
def condition (A C : ℝ) : Prop :=
  sin A * sin C > cos A * cos C

-- Theorem statement
theorem indeterminate_triangle (A B C : ℝ) (h_triangle : is_triangle_ABC A B C) (h_cond : condition A C) :
  ∃ ang : ℝ, ang = B := sorry

end indeterminate_triangle_l411_411389


namespace product_of_ten_consecutive_three_digit_numbers_l411_411474

theorem product_of_ten_consecutive_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n + 9 ≤ 999) :
  ∃ (p : Finset ℕ), p.card ≤ 23 ∧ ∀ i ∈ Finset.range 10, 
  (PrimeFactorization (n + i)).support ⊆ p :=
by sorry

end product_of_ten_consecutive_three_digit_numbers_l411_411474


namespace sum_log_geom_seq_l411_411031

theorem sum_log_geom_seq :
  ∀ (a_n : ℕ → ℝ), 
  (∃ r : ℝ, (∀ n : ℕ, a_n (n + 1) = a_n n * r) ∧ a_n 4 = 2 ∧ a_n 5 = 4) → 
  (∑ i in Finset.range 8, Real.log (a_n i)) = 12 * Real.log 2 :=
by
  intros a_n h
  sorry

end sum_log_geom_seq_l411_411031


namespace ryan_commuting_time_eq_315_l411_411947

def biking_time : Nat := 30 + 40
def bus_time : Nat := 40 + 40 + 50
def friends_ride_time : Nat := 25
def walking_time : Nat := 90

def total_commuting_time : Nat := biking_time + bus_time + friends_ride_time + walking_time

theorem ryan_commuting_time_eq_315 : total_commuting_time = 315 := by
  unfold total_commuting_time biking_time bus_time friends_ride_time walking_time
  rw [Nat.add_comm, Nat.add_assoc]
  exact rfl -- Ryan spends 315 minutes commuting to work every week.

end ryan_commuting_time_eq_315_l411_411947


namespace least_largest_number_of_five_distinct_factors_l411_411136

noncomputable def factors := Set ℕ

theorem least_largest_number_of_five_distinct_factors :
  (∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
   a * b * c * d * e = 55 * 60 * 65 ∧
   ∀ (a' b' c' d' e' : ℕ), a' * b' * c' * d' * e' = 55 * 60 * 65 → a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' → b' ≠ c' → b' ≠ d' → b' ≠ e' → c' ≠ d' → c' ≠ e' → d' ≠ e' → max a' (max b' (max c' (max d' e'))) ≥ max a (max b (max c (max d e))))) :=
sorry

end least_largest_number_of_five_distinct_factors_l411_411136


namespace nickys_pace_l411_411072

theorem nickys_pace (distance : ℝ) (head_start_time : ℝ) (cristina_pace : ℝ) 
    (time_before_catchup : ℝ) (nicky_distance : ℝ) :
    distance = 100 ∧ head_start_time = 12 ∧ cristina_pace = 5 
    ∧ time_before_catchup = 30 ∧ nicky_distance = 90 →
    nicky_distance / time_before_catchup = 3 :=
by
  sorry

end nickys_pace_l411_411072


namespace inequality_proof_l411_411429

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i)
  (h_sum_eq_sq_sum : ∑ i, a i = ∑ i, (a i)^2) :
  ∑ i j in Finset.offDiag (Finset.range n), a i * a j * (1 - a i * a j) ≥ 0 := 
sorry

end inequality_proof_l411_411429


namespace probability_two_dice_show_1_l411_411580

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411580


namespace part_a_part_b_l411_411052

section part_a

variables {A : Matrix (Fin 3) (Fin 3) ℂ}

-- Define the conditions
def elements_have_modulus_one (A : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  ∀ (i j : Fin 3), complex.abs (A i j) = 1

def non_invertible (A : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  A.det = 0

-- Lean statement for part (a)
theorem part_a (hA_modulus : elements_have_modulus_one A) (hA_non_inv : non_invertible A) :
  ∃ i j, i ≠ j ∧ (A i) = (A j) ∨ ∃ i j, i ≠ j ∧ (λ k, A k i) = (λ k, A k j) := 
sorry

end part_a

section part_b

variables {B : Matrix (Fin 4) (Fin 4) ℂ}

-- Define the conditions
def elements_have_modulus_one_b (B : Matrix (Fin 4) (Fin 4) ℂ) : Prop :=
  ∀ (i j : Fin 4), complex.abs (B i j) = 1

def non_invertible_b (B : Matrix (Fin 4) (Fin 4) ℂ) : Prop :=
  B.det = 0

-- Lean statement for part (b)
theorem part_b (hB_modulus : elements_have_modulus_one_b B) (hB_non_inv : non_invertible_b B) :
  ¬ ∃ i j, i ≠ j ∧ (B i) = (B j) ∨ ∃ i j, i ≠ j ∧ (λ k, B k i) = (λ k, B k j) :=
sorry

end part_b

end part_a_part_b_l411_411052


namespace floor_sum_log2_1024_l411_411787

noncomputable def floor_sum_log2 (n : ℕ) : ℕ :=
∑ i in Finset.range (n + 1), Nat.log2 i

theorem floor_sum_log2_1024 : floor_sum_log2 1024 = 8204 := by
  sorry

end floor_sum_log2_1024_l411_411787


namespace sufficient_but_not_necessary_l411_411778

-- Two planes α and β
variables {α β : Plane}

-- A line a such that a ⊆ α
variables {a : Line} (ha : a ⊆ α)

-- The line a is perpendicular to the plane β
variables (h1 : Perpendicular a β)

-- The statement that "the line a is perpendicular to the plane β is a sufficient but not necessary condition for the plane α is perpendicular to the plane β"
theorem sufficient_but_not_necessary : 
  (Perpendicular a β) → (Perpendicular α β) ∧ ¬ (Perpendicular α β → Perpendicular a β) := by
  sorry

end sufficient_but_not_necessary_l411_411778


namespace pair_C_product_not_36_l411_411991

-- Definitions of the pairs
def pair_A : ℤ × ℤ := (-4, -9)
def pair_B : ℤ × ℤ := (-3, -12)
def pair_C : ℚ × ℚ := (1/2, -72)
def pair_D : ℤ × ℤ := (1, 36)
def pair_E : ℚ × ℚ := (3/2, 24)

-- Mathematical statement for the proof problem
theorem pair_C_product_not_36 :
  pair_C.fst * pair_C.snd ≠ 36 :=
by
  sorry

end pair_C_product_not_36_l411_411991


namespace total_balloons_is_18_l411_411309

-- Define the number of balloons each person has
def Fred_balloons : Nat := 5
def Sam_balloons : Nat := 6
def Mary_balloons : Nat := 7

-- Define the total number of balloons
def total_balloons : Nat := Fred_balloons + Sam_balloons + Mary_balloons

-- The theorem statement to prove
theorem total_balloons_is_18 : total_balloons = 18 := sorry

end total_balloons_is_18_l411_411309


namespace find_f_l411_411377

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l411_411377


namespace sum_of_max_min_f_l411_411341

noncomputable def f (x : ℝ) : ℝ := ((x + 2) ^ 2 + sin x) / (x ^ 2 + 4)

theorem sum_of_max_min_f (a : ℝ) (h_nonneg_a : 0 ≤ a) :
  (set.Icc (-a) a).image f ≠ ∅ →
  let S := set.Icc (-a) a in
  let f_max := ⨆ x ∈ S, f x in
  let f_min := ⨅ x ∈ S, f x in
  f_max + f_min = 2 :=
sorry

end sum_of_max_min_f_l411_411341


namespace trigonometric_expression_evaluation_l411_411701

theorem trigonometric_expression_evaluation :
  let tan30 := (Real.sqrt 3) / 3
  let sin60 := (Real.sqrt 3) / 2
  let cot60 := 1 / (Real.sqrt 3)
  let tan60 := Real.sqrt 3
  let cos45 := (Real.sqrt 2) / 2
  (3 * tan30) / (1 - sin60) + (cot60 + Real.cos (Real.pi * 70 / 180))^0 - tan60 / (cos45^4) = 7 :=
by
  -- This is where the proof would go
  sorry

end trigonometric_expression_evaluation_l411_411701


namespace combined_distance_l411_411417

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l411_411417


namespace probability_exactly_two_ones_l411_411558

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411558


namespace output_value_for_five_l411_411737

-- Definitions and conditions
def A := ℝ
def B := ℝ
def f (x : A) : B := a * x + b

variable (a b : ℝ)

-- Conditions from the problem
axiom h1 : f a b 1 = 3
axiom h2 : f a b 8 = 10

-- Proof statement
theorem output_value_for_five : f a b 5 = 6 :=
sorry

end output_value_for_five_l411_411737


namespace true_propositions_even_l411_411662

-- Definitions of a proposition, its converse, inverse, and contrapositive:
def proposition (P Q : Prop) := P → Q
def contrapositive (P Q : Prop) := ¬ Q → ¬ P
def inverse (P Q : Prop) := ¬ P → ¬ Q
def converse (P Q : Prop) := Q → P

-- Conditions:
axiom prop_and_contrapositive (P Q : Prop) : (proposition P Q) = (contrapositive P Q)
axiom inverse_and_converse (P Q : Prop) : (inverse P Q) = (converse P Q)

-- Prove that the number of true propositions is even
theorem true_propositions_even (P Q : Prop) :
  (proposition P Q = true) → 
  (contrapositive P Q = true) → 
  (inverse P Q = true) → 
  (converse P Q = true) → 
  (nat.even (nat.zero)), 
  := sorry

end true_propositions_even_l411_411662


namespace probability_two_dice_show_1_l411_411577

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411577


namespace evaluate_f_difference_l411_411434

def f(x : ℝ) : ℝ := x^5 + x^4 + x^3 + 5 * x

theorem evaluate_f_difference : f(5) - f(-5) = 6550 := by
  sorry

end evaluate_f_difference_l411_411434


namespace supplier_B_stats_l411_411644

-- Definitions based on conditions
def supplier_A_data : List ℕ := [72, 73, 74, 75, 76, 78, 79]
def supplier_A_freq : List ℕ := [1, 1, 5, 3, 3, 1, 1]
def supplier_B_data : List ℕ := [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

-- Predefined values for Supplier A
def supplier_A_avg : ℕ := 75
def supplier_A_median : ℕ := 75
def supplier_A_mode : ℕ := 74
def supplier_A_variance : ℕ := 3.07

-- Definition of the problem to be proved
theorem supplier_B_stats :
  let a := (72 + 75 + 72 + 75 + 78 + 77 + 73 + 75 + 76 + 77 + 71 + 78 + 79 + 72 + 75) / 15;
  let b := 75; -- Mode observed directly from the data
  let c := (3 * (72 - 75) ^ 2 + 4 * (75 - 75) ^ 2 + 2 * (78 - 75) ^ 2 +
            2 * (77 - 75) ^ 2 + (73 - 75) ^ 2 + (76 - 75) ^ 2 + 
            (71 - 75) ^ 2 + (79 - 75) ^ 2) / 15;
  a = 75 ∧ b = 75 ∧ c = 6 := 
by 
  sorry

end supplier_B_stats_l411_411644


namespace age_of_B_is_23_l411_411635

-- Definitions of conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 87)
variable (h2 : A + C = 64)

-- Statement of the problem
theorem age_of_B_is_23 : B = 23 :=
by { sorry }

end age_of_B_is_23_l411_411635


namespace angle_ABC_is_45_deg_l411_411704

variables {A B C P Q M N : Type}

-- Conditions translations
variable (AP_eq_AQ : ∀ (A P Q : ℝ), real.dist A P = real.dist A Q)
variable (projection_condition : ∀ (A B M N : ℝ), (real.dist A M)^2 - (real.dist A N)^2 = (real.dist B N)^2 - (real.dist B M)^2)
variable (P_Q_on_halfline_BC : P = Q)
variable (projections : ∀ (A B C P Q M N : ℝ), is_projection P B M ∧ is_projection Q B N)  -- Assuming is_projection is properly defined elsewhere

-- What needs to be proved
theorem angle_ABC_is_45_deg 
  (hAP : AP_eq_AQ A P Q) 
  (hcond : projection_condition A B M N) 
  (h1 : P_Q_on_halfline_BC) 
  (h2 : projections A B C P Q M N):
  ∃ (angle_ABC : ℝ), angle_ABC = 45 :=
sorry

end angle_ABC_is_45_deg_l411_411704


namespace probability_not_above_y_axis_l411_411471

-- Define the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk (-1) 5
def Q := Point.mk 2 (-3)
def R := Point.mk (-5) (-3)
def S := Point.mk (-8) 5

-- Define predicate for being above the y-axis
def is_above_y_axis (p : Point) : Prop := p.y > 0

-- Define the parallelogram region (this is theoretical as defining a whole region 
-- can be complex, but we state the region as a property)
noncomputable def in_region_of_parallelogram (p : Point) : Prop := sorry

-- Define the probability calculation statement
theorem probability_not_above_y_axis (p : Point) :
  in_region_of_parallelogram p → ¬is_above_y_axis p := sorry

end probability_not_above_y_axis_l411_411471


namespace tangent_line_at_one_l411_411771

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_one :
  let y := f 1 in
  let tangent_slope := (fun x => 1 + Real.log x) 1 in
  let tangent_point := (1 : ℝ, y) in
  ∀ x : ℝ, (y = 0) → (tangent_slope = 1) →  
            ((tangent_point = (1, 0)) → 
            ((y - tangent_point.2) = tangent_slope * (x - tangent_point.1) → y = x - 1)) :=
by
  intros y tangent_slope tangent_point x hy hs ht heq
  sorry

end tangent_line_at_one_l411_411771


namespace sum_even_three_digit_divisible_by_3_l411_411300

theorem sum_even_three_digit_divisible_by_3 :
  ∑ k in finset.range 150, (102 + k * 6) = 82350 :=
by
  sorry

end sum_even_three_digit_divisible_by_3_l411_411300


namespace prove_a_value_tangent_line_equation_l411_411313

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (x - a)^2

theorem prove_a_value (h : ∀ (a : ℝ), ∃ (a : ℝ), (f 0 a) = ((-1) * (-a-2))) :
a = -2 :=
sorry

noncomputable def f_fixed_a (x : ℝ) : ℝ := (x - 1) * (x + 2)^2

theorem tangent_line_equation :
(∀ P : ℝ × ℝ, P = (-3, -4) →
  ∃ l : ℝ → ℝ, l = 9 * P.1 + 23 →
  P = f_fixed_a (1) = 9x - y + 23) :=
sorry

end prove_a_value_tangent_line_equation_l411_411313


namespace sum_possible_values_Alice_card_l411_411422

open Real

theorem sum_possible_values_Alice_card (y : ℝ) (hy1 : 90 * π / 180 < y) (hy2 : y < 180 * π / 180)
  (h_unique : ∃ (v : ℝ), v = sin y ∧ ¬(v = cos y) ∧ ¬(v = tan y)) : ∑ (v : ℝ) in {sin y}, v = 1 / 2 :=
  by 
    sorry

end sum_possible_values_Alice_card_l411_411422


namespace parabola_num_xintercepts_l411_411263

-- Defining the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- The main theorem to state: the number of x-intercepts for the parabola is 2.
theorem parabola_num_xintercepts : ∃ (a b : ℝ), parabola a = 0 ∧ parabola b = 0 ∧ a ≠ b :=
by
  sorry

end parabola_num_xintercepts_l411_411263


namespace direct_proportion_function_l411_411625

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 4

-- Direct proportion function definition
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∀ x, f 0 = 0 ∧ (f x) / x = f 1 / 1

-- Prove that fC (x) is the only direct proportion function among the given options
theorem direct_proportion_function :
  is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l411_411625


namespace sin_thirteen_pi_over_six_l411_411278

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l411_411278


namespace complement_of_S_in_U_l411_411450

variable (U : Set ℕ)
variable (S : Set ℕ)

theorem complement_of_S_in_U (hU : U = {1, 2, 3, 4}) (hS : S = {1, 3}) : U \ S = {2, 4} := by
  sorry

end complement_of_S_in_U_l411_411450


namespace f_n_value_l411_411447

noncomputable def f (n : ℕ) : ℕ :=
  2 + (2^4) + (2^7) + (2^10) + ... + (2^(3*n + 10))

theorem f_n_value (n : ℕ) : f n = (2/7) * (8^(n+4) - 1) :=
sorry

end f_n_value_l411_411447


namespace tom_boxes_needed_l411_411948

-- Definitions of given conditions
def room_length : ℕ := 16
def room_width : ℕ := 20
def box_coverage : ℕ := 10
def already_covered : ℕ := 250

-- The total area of the living room
def total_area : ℕ := room_length * room_width

-- The remaining area that needs to be covered
def remaining_area : ℕ := total_area - already_covered

-- The number of boxes required to cover the remaining area
def boxes_needed : ℕ := remaining_area / box_coverage

-- The theorem statement
theorem tom_boxes_needed : boxes_needed = 7 := by
  -- The proof will go here
  sorry

end tom_boxes_needed_l411_411948


namespace smallest_four_digit_divisible_by_4_and_5_l411_411982

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l411_411982


namespace fraction_proof_l411_411200

theorem fraction_proof (x : ℝ) (h₁ : x = 264) (h₂ : 0.75 * x = f * x + 110) : f = 1 / 3 :=
by
  have h : 0.75 * 264 = f * 264 + 110 := by
    rw [h₁] at h₂
    exact h₂
  have : 198 = f * 264 + 110 := by
    norm_num
    exact h
  have : 198 - 110 = f * 264 := by
    linarith
  have : 88 = f * 264 := this
  have : f = 88 / 264 := by
    field_simp
    exact eq_div_of_mul_eq this
  norm_num at this
  exact this

end fraction_proof_l411_411200


namespace vertex_of_quadratic_l411_411113

def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem vertex_of_quadratic :
  let y := -1 * (x + 1)^2 - 8
  (quadratic_vertex (-1) 2 7) = (-1, -8) :=
sorry

end vertex_of_quadratic_l411_411113


namespace geometric_series_ratio_l411_411320

-- Definitions used in Lean 4
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, (q ≠ 1 ∧ ∀ n : ℕ, a (n+1) = q * a n)

def sum_of_first_n_terms_of_geometric_sequence (a : ℕ → ℝ) :=
  λ n, if n = 0 then 0 else a 0 * (1 - (a 1 / a 0)^n) / (1 - a 1 / a 0)

-- Given conditions
variables {a : ℕ → ℝ} (h_geo : is_geometric_sequence a)
variable (h_a6_eq_8a3 : a 6 = 8 * a 3)

-- The goal statement
theorem geometric_series_ratio (S₆ : ℝ) (S₃ : ℝ)
  (h_S₆ : S₆ = sum_of_first_n_terms_of_geometric_sequence a 6)
  (h_S₃ : S₃ = sum_of_first_n_terms_of_geometric_sequence a 3) :
  S₆ / S₃ = 9 :=
sorry

end geometric_series_ratio_l411_411320


namespace socks_pairing_l411_411374

theorem socks_pairing :
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 2
  white_socks * brown_socks + brown_socks * blue_socks + white_socks * blue_socks = 31 :=
by
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 2
  have h1 : white_socks * brown_socks = 5 * 3 := rfl
  have h2 : brown_socks * blue_socks = 3 * 2 := rfl
  have h3 : white_socks * blue_socks = 5 * 2 := rfl
  show 5 * 3 + 3 * 2 + 5 * 2 = 31 from by sorry

end socks_pairing_l411_411374


namespace g_f_neg3_eq_1741_l411_411058

def f (x : ℤ) : ℤ := x^3 - 3
def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg3_eq_1741 : g (f (-3)) = 1741 := 
by 
  sorry

end g_f_neg3_eq_1741_l411_411058


namespace supplier_B_stats_l411_411643

-- Definitions based on conditions
def supplier_A_data : List ℕ := [72, 73, 74, 75, 76, 78, 79]
def supplier_A_freq : List ℕ := [1, 1, 5, 3, 3, 1, 1]
def supplier_B_data : List ℕ := [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

-- Predefined values for Supplier A
def supplier_A_avg : ℕ := 75
def supplier_A_median : ℕ := 75
def supplier_A_mode : ℕ := 74
def supplier_A_variance : ℕ := 3.07

-- Definition of the problem to be proved
theorem supplier_B_stats :
  let a := (72 + 75 + 72 + 75 + 78 + 77 + 73 + 75 + 76 + 77 + 71 + 78 + 79 + 72 + 75) / 15;
  let b := 75; -- Mode observed directly from the data
  let c := (3 * (72 - 75) ^ 2 + 4 * (75 - 75) ^ 2 + 2 * (78 - 75) ^ 2 +
            2 * (77 - 75) ^ 2 + (73 - 75) ^ 2 + (76 - 75) ^ 2 + 
            (71 - 75) ^ 2 + (79 - 75) ^ 2) / 15;
  a = 75 ∧ b = 75 ∧ c = 6 := 
by 
  sorry

end supplier_B_stats_l411_411643


namespace largest_prime_factor_l411_411286

theorem largest_prime_factor : ∀ (x : ℂ), x = 2 * complex.I → (∃ (p : ℕ), nat.prime p ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ (-(x^10 + x^8 + x^6 + x^4 + x^2 + 1).re.nat_abs) → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_l411_411286


namespace largest_s_for_angle_ratio_l411_411432

theorem largest_s_for_angle_ratio (r s : ℕ) (hr : r ≥ 3) (hs : s ≥ 3) (h_angle_ratio : (130 * (r - 2)) * s = (131 * (s - 2)) * r) :
  s ≤ 260 :=
by 
  sorry

end largest_s_for_angle_ratio_l411_411432


namespace transformation_equivalence_l411_411627

theorem transformation_equivalence (a b c : ℝ) (x y : ℝ)
  (hab : a = b) (hxy : x = y) :
  a ≠ b ∨ c ≠ 0 ∨ x ≠ y :=
by
  have stepA : c ≠ 0 → (a / c^2 = b / c^2) := 
    λ hc, eq.div_eq_of_eq_mul (hab.symm ▸ eq_mul_inv_of_ne_zero (pow_ne_zero 2 hc)).symm (mul_comm _ _)
  
  have stepB : a * c = b * c := by rw [hab]
  
  have stepC : (x^2 + 1) ≠ 0 → (a * (x^2 + 1) = b * (x^2 + 1)) ≠ a := 
    λ hx, eq_mul_inv_of_ne_zero hx ▸ hab.symm

  have stepD : x - 3 = y - 3 := by rw [hxy]
  
  exact Or.inl sorry

end transformation_equivalence_l411_411627


namespace min_value_arith_seq_l411_411814

noncomputable def arith_seq_min_value : ℕ := 
  let a (n : ℕ) := 3 + (n - 1) in
  let f (n : ℕ) := n * (2 * a n - 10) ^ 2 in
  ∃ n : ℕ, f n = 0

theorem min_value_arith_seq (a_n : ℕ → ℕ) (f : ℕ → ℕ) (h1 : a_n 3 = 5) (h2 : ∑ i in range 3 + 1, a_n i = a_n 1 * a_n 5) :
  a n = 3 + (n - 1) → 
  f n = n * (2 * a n - 10)^2 → 
  ∃ n, f n = 0 :=
by sorry

end min_value_arith_seq_l411_411814


namespace warings_formula_l411_411065

/-- Statement of the problem conditions -/
variables {a b : ℝ} 

/-- Definition of the sequence s_n -/
def s : ℕ → ℝ
| n := (roots_complex_to_real ((X ^ 2 + a * X + b: polynomial ℝ))).sum (λ x, x ^ n)

/-- Proving the Waring's formula relation -/
theorem warings_formula :
  ∀ n: ℕ, n > 0 → s n / n = ∑ m in (finset.range ((n / 2).nat_div 1)).filter (λ x, x <= n/2),
    (-1)^(n + m) * (n - m - 1)! / (m! * (n - 2 * m)!) * a^(n - 2 * m) * b^m  :=
by sorry

end warings_formula_l411_411065


namespace kindergarten_boys_l411_411510

def boys (B G : ℕ) : Prop := B * 3 = G * 2

theorem kindergarten_boys (G : ℕ) (h1 : G = 18) : ∃ B : ℕ, boys B G ∧ B = 12 :=
by
  use 12
  rw [h1]
  show boys 12 18
  sorry

end kindergarten_boys_l411_411510


namespace gcd_lcm_product_l411_411294

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_product (a b : ℕ) : gcd a b * lcm a b = a * b :=
begin
  apply Nat.gcd_mul_lcm,
end

example : gcd 12 9 * lcm 12 9 = 108 :=
by
  have h : gcd 12 9 * lcm 12 9 = 12 * 9 := gcd_lcm_product 12 9
  rw [show 12 * 9 = 108, by norm_num] at h
  exact h

end gcd_lcm_product_l411_411294


namespace crayons_total_l411_411040

theorem crayons_total (x y z : ℕ) : 
  let michelle_initial := x,
      janet_initial := y,
      michelle_received := z,
      janet_received := z in
  let janet_total_after_receive := janet_initial + janet_received,
      michelle_total := michelle_initial + michelle_received + janet_total_after_receive in
  michelle_total = x + y + 2 * z :=
by
  sorry

end crayons_total_l411_411040


namespace proportion_eq_l411_411692

notation "circle" => set
notation "quad" => list

variables {A B C D E F M S O : Type} -- Points involved
variables (circle_O : circle ℝ) -- Circle on ℝ
variables (quad_ABCD : quad ℝ) -- Quadrilateral inscribed in circle_O
variables (E F M S : Type) -- Specific points on the geometric configuration

-- Conditions
variable (ABCD_in_circle_O : quad_ABCD ⊂ circle_O) -- Quadrilateral inscribed in Circle O
variable (internal_tangent_circle : ∃ I, circle I ⊂ circle_O) -- Existence of an internal tangent circle
variable (diameter_EF : ∃ E F, E F ∈ circle_O ∧ ¬ collinear E F E ∧ (E F) = diameter) -- EF is a diameter
variable (E_on_same_side_as_A_BD : A_on_same_side_of_BD E) -- E and A on the same side
variable (EF_perpendicular_BD : E F ⊥ B D) -- EF perpendicular to BD
variable (BD_intersects_EF_at_M : BD ⋂ EF = M) -- BD intersects EF at M
variable (BD_intersects_AC_at_S : BD ⋂ AC = S) -- BD intersects AC at S

-- Goal
theorem proportion_eq {A B C D E F M S : Type} (circle_O : circle ℝ) 
    (quad_ABCD : quad ℝ) (ABCD_in_circle_O : quad_ABCD ⊂ circle_O) 
    (internal_tangent_circle : ∃ I, circle I ⊂ circle_O) 
    (diameter_EF : ∃ E F, E F ∈ circle_O ∧ ¬ collinear E F E ∧ (E F) = diameter) 
    (E_on_same_side_as_A_BD : A_on_same_side_of_BD E)
    (EF_perpendicular_BD : E F ⊥ B D) 
    (BD_intersects_EF_at_M : BD ⋂ EF = M) 
    (BD_intersects_AC_at_S : BD ⋂ AC = S) : 
    (AS / SC) = (EM / MF) :=
sorry

end proportion_eq_l411_411692


namespace mangoes_total_l411_411682

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l411_411682


namespace evaluate_expression_l411_411273

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 :=
by sorry

end evaluate_expression_l411_411273


namespace probability_two_math_books_l411_411073

theorem probability_two_math_books (total_books math_books : ℕ) (total_books_eq : total_books = 5) (math_books_eq : math_books = 3) : 
  (nat.choose math_books 2 : ℚ) / (nat.choose total_books 2) = 3 / 10 := by
  rw [total_books_eq, math_books_eq]
  sorry

end probability_two_math_books_l411_411073


namespace complex_number_properties_l411_411761

theorem complex_number_properties :
  let z : ℂ := 1 + complex.i in
  (conj z = 1 - complex.i) ∧
  (z.im = 1) ∧
  (conj z / z = -complex.i) ∧
  (∀ z0 : ℂ, abs (z0 - z) = 1 → abs z0 ≤ real.sqrt 2 + 1) :=
by
  sorry

end complex_number_properties_l411_411761


namespace smallest_four_digit_divisible_by_4_and_5_l411_411983

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l411_411983


namespace obtuse_triangle_longest_side_range_l411_411818

theorem obtuse_triangle_longest_side_range
  (a b : ℝ)
  (ha : a = 1)
  (hb : b = 2)
  (h_obtuse : ∃ C : ℝ, C > 90) :
  ∀ (c : ℝ), (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (c > max a b) → sqrt 5 < c ∧ c < 3 :=
by 
  sorry

end obtuse_triangle_longest_side_range_l411_411818


namespace log_x_gt_log_y_l411_411739

theorem log_x_gt_log_y (a x y : ℝ) (h1 : 1 < a) (h2 : 0 < x) (h3 : x < y) (h4 : y < 1) : log x a > log y a := 
sorry

end log_x_gt_log_y_l411_411739


namespace number_of_sets_M_l411_411923

open Set

noncomputable def M := { s : Set ℕ | ( {a, b} ⊂ s ∧ s ⊂ {a, b, c, d, e} ) }

theorem number_of_sets_M : (∀ (a b c d e : ℕ), Finset.card M = 6) :=
by
  sorry

end number_of_sets_M_l411_411923


namespace sum_factorials_last_two_digits_l411_411974

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l411_411974


namespace one_hundred_fifty_sixth_digit_of_37_over_740_l411_411172

theorem one_hundred_fifty_sixth_digit_of_37_over_740 :
  (λ (n : ℕ), (n ≥ 3) → (Real.frac (37 / 740) * 10^n).floor % 10 = 0) :=
by
  sorry

end one_hundred_fifty_sixth_digit_of_37_over_740_l411_411172


namespace retirement_savings_l411_411080

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l411_411080


namespace rectangular_solid_diagonal_length_l411_411140

theorem rectangular_solid_diagonal_length
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 34)
  (h2 : 4 * (a + b + c) = 28) :
  sqrt (a^2 + b^2 + c^2) = sqrt 15 :=
by
  sorry

end rectangular_solid_diagonal_length_l411_411140


namespace length_of_QS_l411_411038

theorem length_of_QS (P Q R S : Point) 
  (hPQ : dist P Q = 5) 
  (hQR : dist Q R = 12) 
  (hPR : dist P R = 13) 
  (hAngleBisector : angle_bisector P Q R S) : 
  dist Q S = 5 * Real.sqrt 13 / 3 :=
sorry

end length_of_QS_l411_411038


namespace son_work_time_l411_411190

theorem son_work_time (M S : ℝ) 
  (hM : M = 1 / 4)
  (hCombined : M + S = 1 / 3) : 
  S = 1 / 12 :=
by
  sorry

end son_work_time_l411_411190


namespace find_marksman_hit_rate_l411_411655

-- Define the conditions
def independent_shots (p : ℝ) (n : ℕ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ (n ≥ 1)

def hit_probability (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p) ^ n

-- Stating the proof problem in Lean
theorem find_marksman_hit_rate (p : ℝ) (n : ℕ) 
  (h_independent : independent_shots p n) 
  (h_prob : hit_probability p n = 80 / 81) : 
  p = 2 / 3 :=
sorry

end find_marksman_hit_rate_l411_411655


namespace bodyweight_before_training_l411_411247

-- Definition of initial conditions and problem parameters
variables (BW : ℕ)
constant initial_total : ℕ := 2200
constant gain_on_total : ℕ := 0.15 * initial_total
constant gain_on_bodyweight : ℕ := 8
constant final_ratio : ℕ := 10
constant new_total : ℕ := initial_total + gain_on_total

-- Problem statement
theorem bodyweight_before_training : 
  (new_total = final_ratio * (BW + gain_on_bodyweight)) → BW = 245 :=
by
  -- The actual proof is omitted.
  sorry

end bodyweight_before_training_l411_411247


namespace tetrahedron_inequality_1_tetrahedron_inequality_2_l411_411086

variables {A B C D M : Point}
variables {R_A R_B R_C R_D : ℝ} -- R_A, R_B, R_C, R_D are distances from M to vertices A, B, C, D
variables {Δ Π : ℝ} -- Δ is the area of the largest face; Π is the perimeter of the tetrahedron around point M

-- First inequality
theorem tetrahedron_inequality_1 (h1: R_A = distance M A) (h2: R_B = distance M B) (h3: R_C = distance M C) 
  (h4: R_D = distance M D) (h5: Δ = largest_face_area A B C D) :
  R_A^2 + R_B^2 + R_C^2 + R_D^2 ≥ (sqrt 3 / 2) * Δ :=
  sorry

-- Second inequality
theorem tetrahedron_inequality_2 (h1: R_A = distance M A) (h2: R_B = distance M B) (h3: R_C = distance M C)
  (h4: R_D = distance M D) (h5: Π = perimeter_around_point M A B C D) :
  R_A^2 + R_B^2 + R_C^2 + R_D^2 ≥ (1 / 6) * Π^2 :=
  sorry

end tetrahedron_inequality_1_tetrahedron_inequality_2_l411_411086


namespace highest_power_of_3_divides_N_l411_411913

-- Define the range of two-digit numbers and the concatenation function
def concatTwoDigitIntegers : ℕ := sorry  -- Placeholder for the concatenation implementation

-- Integer N formed by concatenating integers from 31 to 68
def N := concatTwoDigitIntegers

-- The statement proving the highest power of 3 dividing N is 3^1
theorem highest_power_of_3_divides_N :
  (∃ k : ℕ, 3^k ∣ N ∧ ¬ 3^(k+1) ∣ N) ∧ 3^1 ∣ N ∧ ¬ 3^2 ∣ N :=
by
  sorry  -- Placeholder for the proof

end highest_power_of_3_divides_N_l411_411913


namespace intersection_points_of_parallel_lines_and_ellipse_l411_411006

noncomputable def parallel_lines_and_ellipse_intersections (L₁ L₂ : ℝ → Prop) : Prop :=
∀ (L₁ L₂ : ℝ → Prop),
  (∀ x y₁ y₂, L₁ x ↔ y₁ = x / 2 ∧ L₂ x ↔ y₂ = x / 2) →
  (¬ ∃ t₁ t₂, (t₁, 0) ∈ L₁ ∧ (t₂, 0) ∈ L₂) →
  (∀ k₁ k₂, k₁ ≠ k₂ → k₁ ≠ 0 ∧ k₂ ≠ 0 → 
    (-2 ≤ 1 - k₁² ∧ 1 - k₁² ≤ 1) ∨ (-2 ≤ 1 - k₂² ∧ 1 - k₂² ≤ 1) ∨ 
    (-2 ≤ 1 - k₁² ∧ 1 - k₂² ≤ 1)) →
  (∃ x₁ x₂, (x₁, 0) ∉ L₁ ∧ (x₂, 0) ∉ L₂) →
  ∃ n, n = 0 ∨ n = 2 ∨ n = 4

theorem intersection_points_of_parallel_lines_and_ellipse (L₁ L₂ : ℝ → Prop) :
  parallel_lines_and_ellipse_intersections L₁ L₂ := sorry
 
end intersection_points_of_parallel_lines_and_ellipse_l411_411006


namespace pascal_logarithm_l411_411435

noncomputable def pascalProduct (n : ℕ) : ℝ :=
  ∏ k in Finset.range(n+1), real.log10 (nat.choose n k)

theorem pascal_logarithm (n : ℕ) : 
  (pascalProduct n) / (real.log10 2) = ((n + 1) * real.log10 (nat.factorial n) - 2 * ∑ k in Finset.range(n+1), real.log10 (nat.factorial k)) / (real.log10 2) :=
sorry

end pascal_logarithm_l411_411435


namespace DC_passes_midpoint_l411_411053

open EuclideanGeometry

variables {A B C D O H M : Point}
variables (semicircle : Circle)
variables (ABC_triangle : Triangle)
variables (altitude_AH : Line)
variables (midpoint_M : Point)
variables (D_tangent : Line)

-- Definitions of the geometrical conditions
axiom inscribed_triangle : semicircle.inscribed_triangle ABC_triangle
axiom center_diameter : semicircle.center = O ∧ semicircle.diameter = segment B C
axiom tangent_lines_intersect : is_tangent_at semicircle A D_tangent ∧ is_tangent_at semicircle B D_tangent ∧ intersection_at D_tangent D

-- The altitude from A in triangle ABC
axiom altitude_definition : altitude_AH = Line.mk A H ∧ perpendicular altitude_AH (Line.mk B C) ∧ intersection_at altitude_AH (Line.mk B C) H

-- Definition of midpoint
axiom midpoint_def : midpoint M A H

-- The theorem to prove
theorem DC_passes_midpoint (h1 : inscribed_triangle semicircle ABC_triangle)
(h2 : center_diameter semicircle)
(h3 : tangent_lines_intersect)
(h4 : altitude_definition)
(h5 : midpoint_def) : lies_on (segment D C) M := sorry

end DC_passes_midpoint_l411_411053


namespace sequence_properties_l411_411439

variables (x z : ℝ)
variables (x_pos : 0 < x) (z_pos : 0 < z) (x_gt_z : x > z)

noncomputable def A : ℕ → ℝ
| 0     := (x + z) / 2
| (n+1) := (A n + H n) / 2

noncomputable def G : ℕ → ℝ
| 0     := real.sqrt (x * z)
| (n+1) := G n

noncomputable def H : ℕ → ℝ
| 0     := 2 * x * z / (x + z)
| (n+1) := 2 / ((1 / A n) + (1 / H n))

theorem sequence_properties :
  (∀ n, A (n+1) < A n) ∧ (∀ n, G (n+1) = G n) ∧ (∀ n, H (n+1) > H n) :=
by
  sorry

end sequence_properties_l411_411439


namespace TotalToysIsNinetyNine_l411_411363

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l411_411363


namespace proof_of_inequality_l411_411002

theorem proof_of_inequality (a : ℝ) (h : (∃ x : ℝ, x - 2 * a + 4 = 0 ∧ x < 0)) :
  (a - 3) * (a - 4) > 0 :=
by
  sorry

end proof_of_inequality_l411_411002


namespace find_ordered_pair_l411_411502

theorem find_ordered_pair (s l : ℝ) :
  (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) →
  (s = -19 ∧ l = -7 / 2) :=
by
  intro h
  have : (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) := h
  sorry

end find_ordered_pair_l411_411502


namespace oliver_shelves_needed_l411_411460

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l411_411460


namespace answer_l411_411386

variable {a : ℕ → ℝ}  -- denoting the geometric sequence by a function from natural numbers to real numbers
variable {r : ℝ}  -- common ratio of the geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

def condition (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 2 * a 4 = 1 / 2

theorem answer (a : ℕ → ℝ) (r : ℝ) [hr : geom_seq a r] (h : condition a r) : a 1 * (a 3) ^ 2 * a 5 = 1 / 4 :=
  sorry

end answer_l411_411386


namespace avg_salary_officers_l411_411819

-- Definitions based on conditions
def avg_salary_all_employees : ℝ := 120
def avg_salary_non_officers : ℝ := 110
def num_officers : ℕ := 15
def num_non_officers : ℕ := 525

-- Problem statement
theorem avg_salary_officers :
  let total_officers_salary := (num_officers:ℝ) * X,
      total_non_officers_salary := (num_non_officers:ℝ) * avg_salary_non_officers,
      total_employees := (num_officers:ℝ) + (num_non_officers:ℝ),
      total_salary_all := total_employees * avg_salary_all_employees in
  total_non_officers_salary + total_officers_salary = total_salary_all →
  X = 470 :=
begin
  sorry
end

end avg_salary_officers_l411_411819


namespace yogurt_cost_l411_411160

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l411_411160


namespace max_knights_l411_411074

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l411_411074


namespace tangent_line_eqn_l411_411284

open Real

noncomputable def f (x : ℝ) : ℝ := x * exp x + 2

theorem tangent_line_eqn :
  let m := (0 * exp 0 + exp 0) in
  let y0 := f 0 in
  ∃ (a b : ℝ), (∀ x, a * x + b = x + 2) ∧ (a = m) ∧ (b = y0 - m * 0) :=
by
  sorry

end tangent_line_eqn_l411_411284


namespace inequality_proof_l411_411375

noncomputable def arithmetic_mean (a : Fin 1988 → ℝ) : ℝ :=
  (Finset.sum Finset.univ a) / 1988

theorem inequality_proof (a : Fin 1988 → ℝ) (hpos : ∀ i, 0 < a i)
  (hmean : arithmetic_mean a = 1988) :
  (Real.geom_mean (λ i j, 1 + (a i / a j) : Fin 1988 → Fin 1988 → ℝ) ) ^ (1 : ℝ / 1988)  ≥  2 ^ 1988 :=
by
  sorry

end inequality_proof_l411_411375


namespace increasing_branch_of_inverse_proportion_l411_411025

-- Define the inverse proportion function and the conditions
def inverse_proportion (k x : ℝ) : ℝ := k / x

-- The Lean theorem statement
theorem increasing_branch_of_inverse_proportion (k : ℝ) (h_knz : k ≠ 0) (h_neg : k < 0) :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ inverse_proportion k x1 < inverse_proportion k x2 :=
sorry

end increasing_branch_of_inverse_proportion_l411_411025


namespace combined_distance_l411_411416

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l411_411416


namespace letters_into_mailboxes_l411_411368

theorem letters_into_mailboxes (letters : ℕ) (mailboxes : ℕ) (h_letters: letters = 3) (h_mailboxes: mailboxes = 4) :
  (mailboxes ^ letters) = 64 := by
  sorry

end letters_into_mailboxes_l411_411368


namespace alligators_after_one_year_l411_411941

noncomputable def alligator_growth (initial_population : ℕ) (growth_factor : ℚ) : ℕ :=
  let final_population := initial_population * (growth_factor ^ 12)
  (final_population : ℚ).toNat

theorem alligators_after_one_year:
  alligator_growth 4 1.5 ≈ 519 := by
  sorry

end alligators_after_one_year_l411_411941


namespace g_of_minus_3_l411_411102

noncomputable def f (x : ℝ) : ℝ := 4 * x - 7
noncomputable def g (y : ℝ) : ℝ := 3 * ((y + 7) / 4) ^ 2 + 4 * ((y + 7) / 4) + 1

theorem g_of_minus_3 : g (-3) = 8 :=
by
  sorry

end g_of_minus_3_l411_411102


namespace cubic_roots_c_over_d_l411_411935

theorem cubic_roots_c_over_d (a b c d : ℤ) (h : a ≠ 0)
  (h_roots : ∃ r1 r2 r3, r1 = -1 ∧ r2 = 3 ∧ r3 = 4 ∧ 
              a * r1 * r2 * r3 + b * (r1 * r2 + r2 * r3 + r3 * r1) + c * (r1 + r2 + r3) + d = 0)
  : (c : ℚ) / d = 5 / 12 := 
sorry

end cubic_roots_c_over_d_l411_411935


namespace solve_equation_l411_411479

theorem solve_equation :
  ∀ x : ℝ, -x^2 = (5*x - 2)/(x + 5) ↔ x = -2 ∨ x = (-3 + sqrt 5)/2 ∨ x = (-3 - sqrt 5)/2 :=
by
  intro x
  sorry

end solve_equation_l411_411479


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411539

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411539


namespace last_two_digits_of_sum_l411_411959

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l411_411959


namespace part_a_part_b_l411_411993

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k  

-- Statement for Part (a)
theorem part_a (n : ℕ) :
  (∑ k in (finset.range (n.div 2 + 1)).filter (λ k, k % 2 = 0), (-1)^(k/2) * binom n k) = 2^(n/2) * real.cos (n * real.pi / 4) := 
sorry

-- Statement for Part (b)
theorem part_b (n : ℕ) :
  (∑ k in (finset.range (n.div 2 + 1)).filter (λ k, k % 2 = 1), (-1)^((k-1)/2) * binom n k) = 2^(n/2) * real.sin (n * real.pi / 4) := 
sorry

end part_a_part_b_l411_411993


namespace area_increase_percentage_area_percentage_increase_length_to_width_ratio_l411_411501

open Real

-- Part (a)
theorem area_increase_percentage (a b : ℝ) :
  (1.12 * a) * (1.15 * b) = 1.288 * (a * b) :=
  sorry

theorem area_percentage_increase (a b : ℝ) :
  ((1.12 * a) * (1.15 * b)) / (a * b) = 1.288 :=
  sorry

-- Part (b)
theorem length_to_width_ratio (a b : ℝ) (h : 2 * ((1.12 * a) + (1.15 * b)) = 1.13 * 2 * (a + b)) :
  a = 2 * b :=
  sorry

end area_increase_percentage_area_percentage_increase_length_to_width_ratio_l411_411501


namespace candies_per_person_l411_411938

theorem candies_per_person (a b people total_candies candies_per_person : ℕ)
  (h1: a = 17)
  (h2: b = 19)
  (h3: people = 9)
  (h4: total_candies = a + b)
  (h5: candies_per_person = total_candies / people) :
  candies_per_person = 4 :=
by sorry

end candies_per_person_l411_411938


namespace product_lcm_gcd_eq_108_l411_411292

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem product_lcm_gcd_eq_108 (a b : ℕ) (h1 : a = 12) (h2 : b = 9) :
  (lcm a b) * (Nat.gcd a b) = 108 := by
  rw [h1, h2] -- replace a and b with 12 and 9
  have lcm_12_9 : lcm 12 9 = 36 := sorry -- find the LCM of 12 and 9
  have gcd_12_9 : Nat.gcd 12 9 = 3 := sorry -- find the GCD of 12 and 9
  rw [lcm_12_9, gcd_12_9]
  norm_num -- simplifies the multiplication
  exact eq.refl 108

end product_lcm_gcd_eq_108_l411_411292


namespace boys_in_class_l411_411108

theorem boys_in_class (n : ℕ) 
  (avg_height : ℕ → ℕ)
  (wrong_height : ℕ → ℕ)
  (actual_height : ℕ → ℕ)
  (corrected_avg_height : ℕ) 
  (avg_height n = 180) 
  (wrong_height n = 166) 
  (actual_height n = 106)
  (corrected_avg_height = 178) :
  n = 30 := 
by
  sorry

end boys_in_class_l411_411108


namespace min_value_of_m_l411_411431

theorem min_value_of_m (n : ℕ) (hn : n ≥ 2) :
  ∃ (a : fin n → ℕ),
    (∀ i, a i < a (i + 1)) ∧
    (∀ i, is_square ((a i)^2 + (a (i + 1))^2)) ∧
    a (n - 1) = 2 * n^2 - 1 :=
by
  sorry

end min_value_of_m_l411_411431


namespace last_two_digits_of_sum_of_first_50_factorials_l411_411961

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l411_411961


namespace greenTeaPriceDecrease_l411_411115

-- Define the conditions as Lean definitions
def priceJune (C : ℝ) := C -- cost per pound of green tea and coffee in June
def priceCoffeeJuly (C : ℝ) := 2 * C -- price of coffee per pound in July
def priceGreenTeaJuly : ℝ := 0.1 -- price of green tea per pound in July
def costMixJuly : ℝ := 3.15 -- cost of 3 lbs mixture of green tea and coffee in July

-- Define the cost per pound of mixture in July
def pricePerPoundMixJuly : ℝ := costMixJuly / 3

-- Equation for the total cost of the mixture using equal quantities
def equation (C : ℝ) := 0.15 + 1.5 * priceCoffeeJuly C = costMixJuly

-- Define the percentage decrease formula
def percentageDecrease (oldPrice newPrice : ℝ) : ℝ := ((oldPrice - newPrice) / oldPrice) * 100

-- The theorem to prove the percentage decrease in the price of green tea
theorem greenTeaPriceDecrease :
  ∀ (C : ℝ), equation C → percentageDecrease (priceJune C) priceGreenTeaJuly = 90 :=
by
  -- The proof would go here
  sorry

end greenTeaPriceDecrease_l411_411115


namespace correlation_relationship_l411_411825

variable (A : Prop) (B : Prop) (C : Prop) (D : Prop)

-- Definitions of the relationships
def volume_edge_relationship : Prop := ¬ A
def angle_sine_relationship : Prop := ¬ B
def sunlight_yield_relationship : Prop := C
def height_vision_relationship : Prop := ¬ D

-- Problem statement: Proving that option C is the correlation relationship
theorem correlation_relationship :
  (¬ A) ∧ (¬ B) ∧ C ∧ (¬ D) → C := by
  intro h
  cases h with v1 h1
  cases h1 with v2 h2
  cases h2 with s1 s2
  exact s1

end correlation_relationship_l411_411825


namespace gumballs_per_package_l411_411361

theorem gumballs_per_package (total_gumballs : ℕ) (packages : ℝ) (h1 : total_gumballs = 100) (h2 : packages = 20.0) :
  total_gumballs / packages = 5 :=
by sorry

end gumballs_per_package_l411_411361


namespace simplify_and_rationalize_l411_411897

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411897


namespace effective_annual_interest_rate_l411_411711

def quarterly_to_annually_compounded_rate (quarterly_rate : ℝ) (compounding_periods : ℕ) : ℝ := 
  (1 + quarterly_rate) ^ compounding_periods - 1

theorem effective_annual_interest_rate
  (annual_rate : ℝ)
  (quarterly_rate := annual_rate / 4)
  (compounding_periods := 4)
  (effective_annual_rate := (quarterly_to_annually_compounded_rate (quarterly_rate / 100) compounding_periods) * 100) :
  annual_rate = 8 → 
  round (effective_annual_rate * 100) / 100 = 8.24 :=
by
  sorry

end effective_annual_interest_rate_l411_411711


namespace last_two_digits_of_sum_l411_411957

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l411_411957


namespace number_of_correct_propositions_is_three_l411_411147

-- Definition of the propositions
def prop1 := ∀ {α : Type} (P : α) (l1 l2 : α), l1 ∈ perpendicular P → l2 ∈ perpendicular P → parallel l1 l2
def prop2 := ∀ {α : Type} (l : α) (P : α), skew l P → (∃! Q : α, perpendicular Q P)
def prop3 := ∀ {α : Type} (a b : α), skew a b → ¬perpendicular a b → (∀ P : α, a ∈ P → ¬perpendicular P b)

-- The number of correct propositions
def num_correct_props := 3

-- The theorem to be proved
theorem number_of_correct_propositions_is_three :
  (if prop1 ∧ prop2 ∧ prop3 then 3 else 0) = num_correct_props := 
by
  sorry

end number_of_correct_propositions_is_three_l411_411147


namespace points_in_quadrant_I_l411_411264

theorem points_in_quadrant_I (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → (x > 0) ∧ (y > 0) := by
  sorry

end points_in_quadrant_I_l411_411264


namespace sum_of_roots_l411_411727

theorem sum_of_roots:
  let a := 1
  let b := -2014
  let c := -2015
  (∀ x : ℝ, x^2 - 2014 * x - 2015 = 0) → (- b / a) = 2014 :=
begin
  sorry
end

end sum_of_roots_l411_411727


namespace purple_ring_weight_l411_411424

def orange_ring_weight : ℝ := 0.08
def white_ring_weight : ℝ := 0.42
def total_weight : ℝ := 0.83

theorem purple_ring_weight : 
  ∃ (purple_ring_weight : ℝ), purple_ring_weight = total_weight - (orange_ring_weight + white_ring_weight) := 
  by
  use 0.33
  sorry

end purple_ring_weight_l411_411424


namespace suitable_survey_is_D_not_suitable_survey_A_not_suitable_survey_B_not_suitable_survey_C_l411_411182

-- Define the possible survey options as a datatype
inductive SurveyOption
| A : SurveyOption
| B : SurveyOption
| C : SurveyOption
| D : SurveyOption

-- Define a predicate indicating whether a given survey option is suitable for a census
def suitable_for_census : SurveyOption → Prop
| SurveyOption.A := false
| SurveyOption.B := false
| SurveyOption.C := false
| SurveyOption.D := true

-- The theorem statement
theorem suitable_survey_is_D : suitable_for_census SurveyOption.D :=
by 
  -- proof will go here (but omitting as instructed)
  admit

-- Verifying the other options are not suitable (not required but added for completeness):
theorem not_suitable_survey_A : ¬ suitable_for_census SurveyOption.A :=
by 
  -- proof will go here (but omitting as instructed)
  admit

theorem not_suitable_survey_B : ¬ suitable_for_census SurveyOption.B :=
by 
  -- proof will go here (but omitting as instructed)
  admit

theorem not_suitable_survey_C : ¬ suitable_for_census SurveyOption.C :=
by 
  -- proof will go here (but omitting as instructed)
  admit

#check suitable_survey_is_D
#check not_suitable_survey_A
#check not_suitable_survey_B
#check not_suitable_survey_C

end suitable_survey_is_D_not_suitable_survey_A_not_suitable_survey_B_not_suitable_survey_C_l411_411182


namespace contrapositive_of_squared_sum_eq_zero_l411_411491

theorem contrapositive_of_squared_sum_eq_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_of_squared_sum_eq_zero_l411_411491


namespace valid_subset_count_l411_411775

-- Define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition for subset A of M
def isValidSubset (A : Finset ℕ) : Prop := (A ⊆ M) ∧ (A.sum id = 8)

-- Prove the number of valid subsets
theorem valid_subset_count : (Finset.filter isValidSubset (Finset.powerset M)).card = 6 :=
by
  -- Proof to be provided
  sorry

end valid_subset_count_l411_411775


namespace linear_funcA_non_linear_funcB_non_linear_funcC_linear_funcD_l411_411626

-- Defining each function option
def funcA (x : ℝ) : ℝ := (1 / 2) * x
def funcB (x : ℝ) : ℝ := 4 / x
def funcC (x : ℝ) : ℝ := 2 * x^2 - 1
def funcD (k x : ℝ) : ℝ := k * x - 2

-- Proving linearity or non-linearity of the given functions
theorem linear_funcA : ∀ x : ℝ, ∃ m b : ℝ, funcA x = m * x + b := by
  sorry

theorem non_linear_funcB : ¬( ∀ x : ℝ, ∃ m b : ℝ, funcB x = m * x + b ) := by
  sorry

theorem non_linear_funcC : ¬( ∀ x : ℝ, ∃ m b : ℝ, funcC x = m * x + b ) := by
  sorry

theorem linear_funcD (k : ℝ): ∀ x : ℝ, ∃ m b : ℝ, funcD k x = m * x + b := by
  sorry

end linear_funcA_non_linear_funcB_non_linear_funcC_linear_funcD_l411_411626


namespace oliver_shelves_needed_l411_411459

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l411_411459


namespace additional_paperclips_ratio_l411_411629

-- Let's define the problem setup
def yun_initial_paperclips : ℕ := 20
def yun_lost_paperclips : ℕ := 12
def marion_paperclips : ℕ := 9

-- Define the fraction more than what Yun currently has
def fraction_more_than_yun (f : ℚ) :=
  yun_initial_paperclips - yun_lost_paperclips * f + 7 = marion_paperclips

-- Main theorem to prove the ratio of additional paperclips Marion has compared to Yun
theorem additional_paperclips_ratio :
  ∃ f : ℚ, fraction_more_than_yun f ∧ (marion_paperclips - (yun_initial_paperclips - yun_lost_paperclips) = yun_initial_paperclips - yun_lost_paperclips * f) ∨
  (marion_paperclips - 7) = 2 * (yun_initial_paperclips - yun_lost_paperclips) :=
sorry

# Check if our theorem is valid
# additional_paperclips_ratio

end additional_paperclips_ratio_l411_411629


namespace angle_equality_l411_411399

-- Defining the square ABCD
variables (A B C D K N L M : Type)
variables (a b c : ℝ)

-- Conditions
def is_square (ABCD : Type) [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D A] :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

def points_on_sides (K N : Type) (A B D : Type) := 
  (K ∈ segment A B) ∧ (N ∈ segment A D)

def ak_an_eq_2bk_dn (A K N B D : Type) [has_dist A K] [has_dist A N] [has_dist B K] [has_dist D N] :=
  dist A K - dist A N = 2 * dist B K * dist D N

-- The theorem to prove
theorem angle_equality (ABCD K N L M : Type) [has_dist : Π X Y : Type, dist X Y : ℝ]
  (h1 : is_square ABCD) (h2 : points_on_sides K N A B D) (h3 : ak_an_eq_2bk_dn A K N B D) :
  ∠B C K = ∠D N C ∧ ∠D N C = ∠B A M :=
sorry

end angle_equality_l411_411399


namespace tim_out_of_pocket_cost_l411_411531

noncomputable def totalOutOfPocketCost : ℝ :=
  let mriCost := 1200
  let xrayCost := 500
  let examinationCost := 400 * (45 / 60)
  let feeForBeingSeen := 150
  let consultationFee := 75
  let physicalTherapyCost := 100 * 8
  let totalCostBeforeInsurance := mriCost + xrayCost + examinationCost + feeForBeingSeen + consultationFee + physicalTherapyCost
  let insuranceCoverage := 0.70 * totalCostBeforeInsurance
  let outOfPocketCost := totalCostBeforeInsurance - insuranceCoverage
  outOfPocketCost

theorem tim_out_of_pocket_cost : totalOutOfPocketCost = 907.50 :=
  by
    -- Proof will be provided here
    sorry

end tim_out_of_pocket_cost_l411_411531


namespace prob_two_ones_in_twelve_dice_l411_411569

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411569


namespace time_for_A_to_beat_B_l411_411390

variables (t_B t_A : ℝ) (s_B : ℝ := 8) (d_A d_B : ℝ)
variable (race_distance : ℝ := 1000)

-- Definitions based on conditions
def condition1 : Prop := race_distance = 1000
def condition2 : Prop := d_A = d_B + 200
def condition3 : Prop := s_B = 8
def condition4 : Prop := d_B = s_B * t_B
def condition5 : Prop := d_A = race_distance

-- Theorem statement
theorem time_for_A_to_beat_B
  (h1 : condition1) -- A 1000 meter race
  (h2 : condition2) -- A beats B by 200 meters
  (h3 : condition3) -- The speed of B is 8 m/s
  (h4 : condition4) -- Distance covered by B
  (h5 : condition5) -- Total distance A runs
  : t_B = 100 ∧ t_A = t_B := sorry

end time_for_A_to_beat_B_l411_411390


namespace minimum_balls_for_pockets_aligned_l411_411642

theorem minimum_balls_for_pockets_aligned (n : ℕ) :
  (∀ (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1),
    (∃ (a b c d : fin n), 
      (pocket_in_line a b (0, 0) (2, 0) ∨ 
       pocket_in_line a b (0, 0) (0, 1) ∨ 
       pocket_in_line a b (2, 0) (2, 1) ∨ 
       pocket_in_line a b (0, 1) (2, 1) ∨ 
       pocket_in_line a b (0, 0) (2, 1) ∨ 
       pocket_in_line a b (2, 0) (0, 1) ∨ 
       pocket_in_line a b (1, 0) (1, 1) ∨ 
       pocket_in_line a b (0, 0) (1, 1) ∨ 
       pocket_in_line a b (1, 0) (0, 1))))
  → n ≥ 4 :=
begin
  sorry
end

end minimum_balls_for_pockets_aligned_l411_411642


namespace general_formula_sum_first_8_terms_b_n_l411_411325

-- Problem Statement:
-- Given an arithmetic sequence {a_n} satisfies a_4 - a_2 = 4 and a_3 = 8.
-- (I) Prove that the general formula for the sequence {a_n} is a_n = 2n + 2.
-- (II) The sequence {b_n} satisfies b_n = (sqrt(2))^{a_n}. Prove that the sum 
-- of the first 8 terms of the sequence {b_n} is 1020.

noncomputable def a_n (n : ℕ) : ℤ := 2 * ↑n + 2

theorem general_formula (a : ℕ → ℤ) (h₁ : a 4 - a 2 = 4) (h₂ : a 3 = 8) :
  a = a_n :=
  sorry

noncomputable def b_n (n : ℕ) : ℝ := (2 : ℝ) ^ (n + 1)

theorem sum_first_8_terms_b_n :
  (∑ i in Finset.range 8, b_n i) = 1020 :=
  sorry

end general_formula_sum_first_8_terms_b_n_l411_411325


namespace geom_seq_ratio_l411_411057

noncomputable def geom_seq_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_ratio
  (a₁ q : ℝ) (h : q ≠ 1) :
  (geom_seq_sum a₁ q 8 / geom_seq_sum a₁ q 4 = 4) →
  (geom_seq_sum a₁ q 16 / geom_seq_sum a₁ q 8 = 10) :=
begin
  sorry -- proof is omitted
end

end geom_seq_ratio_l411_411057


namespace positive_difference_in_height_of_crates_l411_411611

theorem positive_difference_in_height_of_crates :
    ∀ (d : ℕ) (n : ℕ) (height_A height_B : ℕ),
      (d = 12) →
      (n = 300) →
      (∃ k, k * (k + 1) / 2 = n ∧ height_A = k * d) →
      (∃ k, k^2 = n ∧ height_B = k * d) →
      abs (height_A - height_B) = 72 :=
begin
  intros d n height_A height_B h_d h_n h_A h_B,
  sorry
end

end positive_difference_in_height_of_crates_l411_411611


namespace min_AD_AB_in_triangle_l411_411039

theorem min_AD_AB_in_triangle (A B C D : Point) (h_angle_A : ∠A = 2 * Real.pi / 3)
  (h_D_on_BC : D ∈ Segment B C) (h_2BD_eq_DC : 2 * (dist B D) = dist D C)
  : ∃ k, k = (dist A D) / (dist A B) ∧ k = Real.sqrt 3 / 3 := by
  sorry

end min_AD_AB_in_triangle_l411_411039


namespace units_digit_of_35_87_plus_93_53_l411_411192

theorem units_digit_of_35_87_plus_93_53 : 
  let units_digit_of (n : ℕ) := n % 10,
      digit_pat_3 := λ (k : ℕ), [3, 9, 7, 1].get? ((k - 1) % 4),
      u_35 := units_digit_of 35,
      u_93 := units_digit_of 93,
      u_35_87 := 5,  -- by the property of units digit of 5
      u_93_53 := 3  -- by the property of repeating pattern of powers of 3 
  in u_35 = 5 ∧ u_93 = 3 ∧ u_35_87 = 5 ∧ 
     (53 - 1) % 4 + 1 = 1 ∧ digit_pat_3 1 = some 3 ∧ 
     (units_digit_of (35 ^ 87) = u_35_87) ∧ 
     (units_digit_of (93 ^ 53) = u_93_53)
   → units_digit_of ((35) ^ (87) + (93) ^ (53)) = 8 :=
begin
  sorry
end

end units_digit_of_35_87_plus_93_53_l411_411192


namespace complex_division_l411_411111

theorem complex_division : (2 * complex.I) / (1 + complex.I) = 1 + complex.I :=
by
  sorry

end complex_division_l411_411111


namespace ABC_isosceles_l411_411951

-- Declare the problem parameters
variables (C1 C2 : Circle) (P Q A B C : Point)

-- Declare the problem conditions
axiom touch_externally (h : ∃ (O1 O2 : Point), circle_center C1 O1 ∧ circle_center C2 O2 ∧ touches_externally_at C1 C2 P)
axiom Q_on_C1 (hq : on_circle Q C1)
axiom tangent_through_Q (ht : ∃ T : Line, is_tangent T C1 Q ∧ intersects T C2 A B)
axiom QP_intersects_C2_at_C (hi : line_through Q P.intersects_circle C2 C)

-- Declare the conclusion to be proved
theorem ABC_isosceles
  (h: exists (O1 O2 : Point), circle_center C1 O1 ∧ circle_center C2 O2 ∧ touches_externally_at C1 C2 P)
  (hq: on_circle Q C1)
  (ht: exists T : Line, is_tangent T C1 Q ∧ intersects T C2 A B)
  (hi: line_through Q P.intersects_circle C2 C) :
  is_isosceles_triangle A B C :=
sorry

end ABC_isosceles_l411_411951


namespace abs_inequality_solution_set_l411_411514

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 1| + |x + 2| < 5 ↔ -3 < x ∧ x < 2 :=
by {
  sorry
}

end abs_inequality_solution_set_l411_411514


namespace rowing_downstream_speed_l411_411654

-- Define the given conditions
def V_u : ℝ := 60  -- speed upstream in kmph
def V_s : ℝ := 75  -- speed in still water in kmph

-- Define the problem statement
theorem rowing_downstream_speed : ∃ (V_d : ℝ), V_s = (V_u + V_d) / 2 ∧ V_d = 90 :=
by
  sorry

end rowing_downstream_speed_l411_411654


namespace prop_p_iff_prop_q_l411_411809

variable {A B C : Angle}
variable {a b c : Real}
variable {tABC : Triangle}
variable {p q : Prop}
variable {equilateralABC : EquilateralTriangle tABC}

-- conditions
def sides_opposite (t : Triangle) := 
  let a := t.side1
  let b := t.side2
  let c := t.side3
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def prop_p (t : Triangle) :=
  let A := t.angleA
  let B := t.angleB
  let C := t.angleC
  (B + C = 2 * A) ∧ (t.side2 + t.side3 = 2 * t.side1)

def prop_q (t : Triangle) :=
  t.is_equilateral

-- The main proof statement
theorem prop_p_iff_prop_q (t : Triangle) : prop_p t ↔ prop_q t := sorry

end prop_p_iff_prop_q_l411_411809


namespace decreasing_implies_bound_l411_411789

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_implies_bound (b : ℝ) :
  (∀ x > 2, -x + b / x ≤ 0) → b ≤ 4 :=
  sorry

end decreasing_implies_bound_l411_411789


namespace simplify_rationalize_denominator_l411_411906

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411906


namespace total_sum_of_intersections_of_five_lines_l411_411009

theorem total_sum_of_intersections_of_five_lines : 
  let possible_intersections : List ℕ := [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
  (possible_intersections.sum = 53) :=
by
  sorry

end total_sum_of_intersections_of_five_lines_l411_411009


namespace result_is_2750_l411_411621

noncomputable def problem_statement : ℝ :=
  let sum1 := ∑ k in Finset.range 10, Real.logb (7^k) (2^(k^2))
  let sum2 := ∑ k in Finset.range 50, Real.logb (4^k) (49^k)
  sum1 * sum2

theorem result_is_2750 : problem_statement = 2750 := by
  sorry

end result_is_2750_l411_411621


namespace irrationa_y_l411_411857

noncomputable def f (x : ℤ) : ℤ

variable (h1 : ∀ x, f (x) ∈ ℤ)
variable (h2 : Polynomial.degree (Polynomial.monic (f x)) ≥ 1)
variable (h3 : ∀ x ≥ 1, f (x) > 0)

theorem irrationa_y : ¬ Rational (0.f(1)f(2)f(3)...) := by
  sorry

end irrationa_y_l411_411857


namespace c_minus_a_is_10_l411_411067

variable (a b c d k : ℝ)

theorem c_minus_a_is_10 (h1 : a + b = 90)
                        (h2 : b + c = 100)
                        (h3 : a + c + d = 180)
                        (h4 : a^2 + b^2 + c^2 + d^2 = k) :
  c - a = 10 :=
by sorry

end c_minus_a_is_10_l411_411067


namespace probability_two_ones_in_twelve_dice_l411_411554

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411554


namespace enrique_shredder_pages_l411_411271

theorem enrique_shredder_pages (total_contracts : ℕ) (num_times : ℕ) (pages_per_time : ℕ) :
  total_contracts = 2132 ∧ num_times = 44 → pages_per_time = 48 :=
by
  intros h
  sorry

end enrique_shredder_pages_l411_411271


namespace tank_empty_time_l411_411211

theorem tank_empty_time 
  (time_to_empty_leak : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (tank_volume : ℝ) 
  (net_time_to_empty : ℝ) : 
  time_to_empty_leak = 7 → 
  inlet_rate_per_minute = 6 → 
  tank_volume = 6048.000000000001 → 
  net_time_to_empty = 12 :=
by
  intros h1 h2 h3
  sorry

end tank_empty_time_l411_411211


namespace coefficient_x_squared_in_expansion_l411_411489

theorem coefficient_x_squared_in_expansion :
  (∃ c : ℤ, (1 + x)^6 * (1 - x) = c * x^2 + b * x + a) → c = 9 :=
by
  sorry

end coefficient_x_squared_in_expansion_l411_411489


namespace angle_AMD_is_70_degrees_l411_411088

noncomputable def right_triangle_angle (AB BC : ℝ) (M: ℝ) : ℝ :=
let AC := Real.sqrt (AB^2 - BC^2) in
let AM := AB / 2 in
let BM := AB / 2 in
let ∠BMC := Real.atan (BC / BM) in
(90 + Real.toDegrees (Real.atan2 BC BM)) / 2

theorem angle_AMD_is_70_degrees 
  (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 6)
  (M : ℝ) (h_AM_CM : 2 * M = AB) : 
  right_triangle_angle AB BC M = 70 :=
by sorry

end angle_AMD_is_70_degrees_l411_411088


namespace find_vec_c_l411_411782

open Real

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (2, -3)
def vec_c : ℝ × ℝ := (-7/9, -7/3) -- solution we aim to prove

lemma vec_c_parallel_b :
  ∃ (x y : ℝ), (x, y) = vec_c ∧ (x + 1, y + 2) = (2, -3) * (some k : ℝ) :=
sorry

lemma vec_c_perpendicular_sum :
  ∃ (x y : ℝ), (x, y) = vec_c ∧ 3 * x - y = 0 :=
sorry

theorem find_vec_c : 
exists (c : ℝ × ℝ), 
  c = vec_c ∧ (c + vec_a ∥ vec_b) ∧ (dot_product c (vec_a + vec_b) = 0) :=
by
  use vec_c
  split
  · exact rfl
  split
  · exact vec_c_parallel_b
  · exact vec_c_perpendicular_sum

end find_vec_c_l411_411782


namespace af_over_at_l411_411413

open_locale classical

-- Definitions of points and segments
variables {A B C D E F T : Type}

-- Given conditions
variables [add_comm_group A] [module ℝ A]
variables (a b c d e f t : A) (AF AT : ℝ)

-- Point D and E segments, based on given conditions
def point_d : A := (2 • a + 2 • b) / 2
def point_e : A := (3 • a + 3 • c) / 2

-- Angle bisector segment equality
def point_t : A := (b + c) / 2

-- Main theorem to prove
theorem af_over_at : (AD : ℝ) = 2 → (DB : ℝ) = 2 → (AE : ℝ) = 3 → (EC : ℝ) = 3 → 
    ∃ (λ: ℝ), ∃ (μ: ℝ), λ = 1 / 2 ∧ μ = 1 / 2 → (AF / AT) = 1 / 2 :=
by sorry

end af_over_at_l411_411413


namespace least_subtracted_correct_second_num_correct_l411_411622

-- Define the given numbers
def given_num : ℕ := 1398
def remainder : ℕ := 5
def num1 : ℕ := 7
def num2 : ℕ := 9
def num3 : ℕ := 11

-- Least number to subtract to satisfy the condition
def least_subtracted : ℕ := 22

-- Second number in the sequence
def second_num : ℕ := 2069

-- Define the hypotheses and statements to be proved
theorem least_subtracted_correct : given_num - least_subtracted ≡ remainder [MOD num1]
∧ given_num - least_subtracted ≡ remainder [MOD num2]
∧ given_num - least_subtracted ≡ remainder [MOD num3] := sorry

theorem second_num_correct : second_num ≡ remainder [MOD num1 * num2 * num3] := sorry

end least_subtracted_correct_second_num_correct_l411_411622


namespace log_limit_l411_411793

theorem log_limit (x : ℝ) (hx : 0 < x) (h : x → ∞) :
  (log 5 (10 * x - 3) - log 5 (4 * x + 7)) → log 5 2.5 :=
by
  sorry

end log_limit_l411_411793


namespace total_volume_of_quiche_l411_411862

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end total_volume_of_quiche_l411_411862


namespace triangle_XYZ_ratio_l411_411013

theorem triangle_XYZ_ratio (XZ YZ : ℝ)
  (hXZ : XZ = 9) (hYZ : YZ = 40)
  (XY : ℝ) (hXY : XY = Real.sqrt (XZ ^ 2 + YZ ^ 2))
  (ZD : ℝ) (hZD : ZD = Real.sqrt (XZ * YZ))
  (XJ YJ : ℝ) (hXJ : XJ = Real.sqrt (XZ * (XZ + 2 * ZD)))
  (hYJ : YJ = Real.sqrt (YZ * (YZ + 2 * ZD)))
  (ratio : ℝ) (h_ratio : ratio = (XJ + YJ + XY) / XY) :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ ratio = p / q ∧ p + q = 203 := sorry

end triangle_XYZ_ratio_l411_411013


namespace find_a_perpendicular_line_l411_411800

theorem find_a_perpendicular_line (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 3 * y + 1 = 0) → (2 * x + 2 * y - 3 = 0) → (-(a / 3) * (-1) = -1)) → 
  a = -3 :=
by
  sorry

end find_a_perpendicular_line_l411_411800


namespace kibble_left_l411_411050

-- Define the initial amount of kibble
def initial_kibble := 3

-- Define the rate at which the cat eats kibble
def kibble_rate := 1 / 4

-- Define the time Kira was away
def time_away := 8

-- Define the amount of kibble eaten by the cat during the time away
def kibble_eaten := (time_away * kibble_rate)

-- Define the remaining kibble in the bowl
def remaining_kibble := initial_kibble - kibble_eaten

-- State and prove that the remaining amount of kibble is 1 pound
theorem kibble_left : remaining_kibble = 1 := by
  sorry

end kibble_left_l411_411050


namespace altitudes_sum_l411_411920

theorem altitudes_sum (x y : ℝ) (h: 9 * x + 7 * y = 63): 
  x = 7 → y = 9 → 7 + 9 + 63 / Real.sqrt 130 = 2695 / 130 := 
by 
  intros hx hy 
  rw [hx, hy]
  sorry

end altitudes_sum_l411_411920


namespace initial_pencils_correct_l411_411528

variable (initial_pencils : ℕ)
variable (pencils_added : ℕ := 45)
variable (total_pencils : ℕ := 72)

theorem initial_pencils_correct (h : total_pencils = initial_pencils + pencils_added) : initial_pencils = 27 := by
  sorry

end initial_pencils_correct_l411_411528


namespace reciprocal_of_neg_one_fifth_l411_411179

theorem reciprocal_of_neg_one_fifth : ∃ b : ℚ, (b * (-1/5) = 1) ∧ (b = -5) :=
by {
  use -5,
  split,
  {
    calc (-5) * (-1/5) = 25/5 : by { simp [mul_neg_eq_neg_mul_symm, div_eq_mul_inv, mul_inv_cancel], ring }
                    ... = 5 : by norm_num,
    linarith,
  },
  refl
}

end reciprocal_of_neg_one_fifth_l411_411179


namespace largest_volume_cuboid_l411_411696

theorem largest_volume_cuboid (a b c : ℝ) (P : ℝ) (h : a * b + b * c + a * c = P) :
  (a = b = c) → ∀ (x y z : ℝ), x * y + y * z + x * z ≤ P → x * y * z ≤ a * b * c :=
by
  sorry

end largest_volume_cuboid_l411_411696


namespace sandy_siding_cost_l411_411089

theorem sandy_siding_cost:
  let wall_width := 8
  let wall_height := 8
  let roof_width := 8
  let roof_height := 5
  let siding_width := 10
  let siding_height := 12
  let siding_cost := 30
  let wall_area := wall_width * wall_height
  let roof_side_area := roof_width * roof_height
  let roof_area := 2 * roof_side_area
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let required_sections := (total_area + siding_area - 1) / siding_area -- ceiling division
  let total_cost := required_sections * siding_cost
  total_cost = 60 :=
by
  sorry

end sandy_siding_cost_l411_411089


namespace general_admission_tickets_sold_l411_411530

theorem general_admission_tickets_sold:
  ∃ (S G : ℕ), 4 * S + 6 * G = 2876 ∧ S + G = 525 ∧ G = 388 :=
by {
  -- Here we assume there exist natural numbers S and G
  exists.intro _ _,
  sorry
}

end general_admission_tickets_sold_l411_411530


namespace find_y_l411_411405

variables (ABC ACB BAC : ℝ)
variables (CDE ADE EAD AED DEB y : ℝ)

-- Conditions
axiom angle_ABC : ABC = 45
axiom angle_ACB : ACB = 90
axiom angle_BAC_eq : BAC = 180 - ABC - ACB
axiom angle_CDE : CDE = 72
axiom angle_ADE_eq : ADE = 180 - CDE
axiom angle_EAD : EAD = 45
axiom angle_AED_eq : AED = 180 - ADE - EAD
axiom angle_DEB_eq : DEB = 180 - AED
axiom y_eq : y = DEB

-- Goal
theorem find_y : y = 153 :=
by {
  -- Here we would proceed with the proof using the established axioms.
  sorry
}

end find_y_l411_411405


namespace find_x_l411_411254

theorem find_x (x : ℝ) (h : (x + 8 + 5 * x + 4 + 2 * x + 7) / 3 = 3 * x - 10) : x = 49 :=
sorry

end find_x_l411_411254


namespace sequence_x_n_l411_411139

theorem sequence_x_n (x : ℕ → ℝ) (h₁ : x 1 = 2) 
  (h₂ : ∀ n > 1, (∑ i in Finset.range (n-1), x (i+1)) + (3/2) * x n = 3) : 
  x 1000 = 2 / 3 ^ 999 :=
sorry

end sequence_x_n_l411_411139


namespace find_k_l411_411335

theorem find_k (k : ℕ) (h_pos : 0 < k) : 
  let coeff := (Nat.factorial 6) / (Nat.factorial 4 * Nat.factorial (6 - 4)) * k^4 in 
  coeff < 120 → k = 1 := 
by
  let coeff := 15 * k^4
  intro h
  have h_coeff : coeff = (Nat.factorial 6) / (Nat.factorial 4 * Nat.factorial (6 - 4)) * k^4 := by
    simp [coeff, Nat.factorial]
  have := (15 : ℕ) * k^4 < 120
  sorry

end find_k_l411_411335


namespace eugene_cards_in_deck_l411_411272

theorem eugene_cards_in_deck 
  (cards_used_per_card : ℕ)
  (boxes_used : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_leftover : ℕ)
  (total_toothpicks_used : ℕ)
  (cards_used : ℕ)
  (total_cards_in_deck : ℕ)
  (h1 : cards_used_per_card = 75)
  (h2 : boxes_used = 6)
  (h3 : toothpicks_per_box = 450)
  (h4 : cards_leftover = 16)
  (h5 : total_toothpicks_used = boxes_used * toothpicks_per_box)
  (h6 : cards_used = total_toothpicks_used / cards_used_per_card)
  (h7 : total_cards_in_deck = cards_used + cards_leftover) :
  total_cards_in_deck = 52 :=
by 
  sorry

end eugene_cards_in_deck_l411_411272


namespace last_two_digits_of_sum_of_first_50_factorials_l411_411963

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l411_411963


namespace volume_frustum_correct_l411_411229

noncomputable def volume_of_frustum 
  (base_edge_orig : ℝ) 
  (altitude_orig : ℝ) 
  (base_edge_small : ℝ) 
  (altitude_small : ℝ) : ℝ :=
  let volume_ratio := (base_edge_small / base_edge_orig) ^ 3
  let base_area_orig := (Real.sqrt 3 / 4) * base_edge_orig ^ 2
  let volume_orig := (1 / 3) * base_area_orig * altitude_orig
  let volume_small := volume_ratio * volume_orig
  let volume_frustum := volume_orig - volume_small
  volume_frustum

theorem volume_frustum_correct :
  volume_of_frustum 18 9 9 3 = 212.625 * Real.sqrt 3 :=
sorry

end volume_frustum_correct_l411_411229


namespace derivative_at_one_l411_411344

theorem derivative_at_one (f : ℝ → ℝ) (df : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2) 
  (h₂ : ∀ x, df x = 2 * x) : 
  df 1 = 2 :=
by sorry

end derivative_at_one_l411_411344


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411538

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411538


namespace quiche_total_volume_l411_411864

theorem quiche_total_volume :
  ∀ (raw_spinach cream_cheese eggs : ℕ),
    raw_spinach = 40 →
    cream_cheese = 6 →
    eggs = 4 →
    let cooked_spinach := raw_spinach * 20 / 100 in
    cooked_spinach + cream_cheese + eggs = 18 :=
by
  intros raw_spinach cream_cheese eggs h_raw h_cream h_eggs
  simp [h_raw, h_cream, h_eggs]
  -- rewriting the calculation step
  let cooked_spinach := raw_spinach * 20 / 100
  have h_cooked : cooked_spinach = 8 := by norm_num
  rw h_cooked
  norm_num
  -- closing the final proof by setting result to 18
  sorry

end quiche_total_volume_l411_411864


namespace ram_account_balance_first_year_l411_411168

theorem ram_account_balance_first_year :
  let initial_deposit := 1000
  let interest_first_year := 100
  initial_deposit + interest_first_year = 1100 :=
by
  sorry

end ram_account_balance_first_year_l411_411168


namespace max_triangles_from_parallel_lines_l411_411150

theorem max_triangles_from_parallel_lines (S1 S2 S3 : Finset (Set ℝ)) 
  (hS1 : S1.card = 10) (hS2 : S2.card = 10) (hS3 : S3.card = 10)
  (h_parallel_S1 : ∀ l1 l2 ∈ S1, Parallel l1 l2)
  (h_parallel_S2 : ∀ l1 l2 ∈ S2, Parallel l1 l2)
  (h_parallel_S3 : ∀ l1 l2 ∈ S3, Parallel l1 l2) :
  ∀ P : Set (Finset (Set ℝ)), (∀ l1 l2 l3 ∈ P, l1 ∉ S1 ∧ l2 ∉ S2 ∧ l3 ∉ S3 ∧ Intersect l1 l2 l3 ∧ P.card = 3) → P.card = 150 := 
sorry

end max_triangles_from_parallel_lines_l411_411150


namespace quiche_total_volume_l411_411863

theorem quiche_total_volume :
  ∀ (raw_spinach cream_cheese eggs : ℕ),
    raw_spinach = 40 →
    cream_cheese = 6 →
    eggs = 4 →
    let cooked_spinach := raw_spinach * 20 / 100 in
    cooked_spinach + cream_cheese + eggs = 18 :=
by
  intros raw_spinach cream_cheese eggs h_raw h_cream h_eggs
  simp [h_raw, h_cream, h_eggs]
  -- rewriting the calculation step
  let cooked_spinach := raw_spinach * 20 / 100
  have h_cooked : cooked_spinach = 8 := by norm_num
  rw h_cooked
  norm_num
  -- closing the final proof by setting result to 18
  sorry

end quiche_total_volume_l411_411863


namespace goal_winning_rate_l411_411628

theorem goal_winning_rate (played : ℕ) (won : ℕ) (add_wins : ℕ) : 
  played = 20 → 
  won = 19 → 
  (won + add_wins) / (played + add_wins) = 0.96 → 
  add_wins = 5 :=
sorry

end goal_winning_rate_l411_411628


namespace prob_two_ones_in_twelve_dice_l411_411572

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411572


namespace min_side_length_is_isosceles_l411_411473

-- Let a denote the side length BC
-- Let b denote the side length AB
-- Let c denote the side length AC

theorem min_side_length_is_isosceles (α : ℝ) (S : ℝ) (a b c : ℝ) :
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos α ∧ S = 0.5 * b * c * Real.sin α) →
  a = Real.sqrt (((b - c)^2 + (4 * S * (1 - Real.cos α)) / Real.sin α)) →
  b = c :=
by
  intros h1 h2
  sorry

end min_side_length_is_isosceles_l411_411473


namespace cylinder_height_l411_411169

theorem cylinder_height
  (r : ℝ) (SA : ℝ) (h : ℝ)
  (h_radius : r = 3)
  (h_surface_area_given : SA = 30 * Real.pi)
  (h_surface_area_formula : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  h = 2 :=
by
  -- Proof can be written here
  sorry

end cylinder_height_l411_411169


namespace high_heels_height_l411_411246

variable (l : ℝ) (r_ideal : ℝ) (x_ratio : ℝ)

theorem high_heels_height (h : l = 160) (hr : r_ideal = 0.618) (hx : x_ratio = 0.60) : 
  let x := l * x_ratio,
      y := (r_ideal * (l + y) - x) / (1 - r_ideal)
  in 
  round y = 8 :=
by
  sorry

end high_heels_height_l411_411246


namespace number_of_mappings_l411_411780

noncomputable def A : Set ℝ := {a1, a2, a3, a4, a5}
noncomputable def B : Set ℝ := {b1, b2, b3, b4, b5}
def f : ℝ → ℝ

axiom f_ordering : f a1 ≥ f a2 ∧ f a2 ≥ f a3 ∧ f a3 ≥ f a4 ∧ f a4 ≥ f a5
axiom exactly_one_no_image : ∃ b ∈ B, ∀ a ∈ A, f a ≠ b

theorem number_of_mappings : 
  let mappings := (4 * Nat.choose 5 4) in
  mappings = 20 := 
by
  sorry

end number_of_mappings_l411_411780


namespace integral_inequality_l411_411104

variable {f : ℝ → ℝ → ℝ}

theorem integral_inequality 
  (h_cont : ∀ x y, continuous (f x y)) :
  ∫ (y : ℝ) in 0..1, (∫ (x : ℝ) in 0..1, f x y) ^ 2 ≤
  (∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in 0..1, f x y) ^ 2 + 
  ∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in 0..1, (f x y) ^ 2 := sorry

end integral_inequality_l411_411104


namespace dice_probability_l411_411599

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411599


namespace combined_distance_l411_411418

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l411_411418


namespace math_inequality_l411_411314

noncomputable def math_problem (n : ℕ) (x : Fin n → ℝ) (m a s : ℝ) :=
  (∀ i, 0 < x i) ∧
  0 < m ∧
  0 < a ∧
  (∑ i, x i = s) ∧
  (s ≤ n) →
  (∑ i, (x i ^ m + x i ^ (-m) + a) ^ n) ≥
  n * ((s / n) ^ m + (n / s) ^ m + a) ^ n

theorem math_inequality (n : ℕ) (x : Fin n → ℝ) (m a s : ℝ) :
  math_problem n x m a s := by
  sorry

end math_inequality_l411_411314


namespace exists_good_pair_for_each_m_l411_411454

def is_good_pair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = a^2 ∧ (m + 1) * (n + 1) = b^2

theorem exists_good_pair_for_each_m : ∀ m : ℕ, ∃ n : ℕ, m < n ∧ is_good_pair m n := by
  intro m
  let n := m * (4 * m + 3)^2
  use n
  have h1 : m < n := sorry -- Proof that m < n
  have h2 : is_good_pair m n := sorry -- Proof that (m, n) is a good pair
  exact ⟨h1, h2⟩

end exists_good_pair_for_each_m_l411_411454


namespace kotelmel_area_error_l411_411388

theorem kotelmel_area_error (a : ℝ) : 
  abs ((sqrt 3 - (26/15)) / sqrt 3) * 100 ≈ 0.075 :=
by
  sorry

end kotelmel_area_error_l411_411388


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411540

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411540


namespace min_socks_to_guarantee_pairs_l411_411649

theorem min_socks_to_guarantee_pairs :
  ∀ (R Y G P : ℕ), R = 50 → Y = 100 → G = 70 → P = 30 →
  ∃ (n : ℕ), (∀ (socks_drawn : list ℕ),
    socks_drawn.length = n →
    (∃ (pairs : ℕ), pairs ≥ 8)) ∧ n = 28 :=
begin
  intros R Y G P hR hY hG hP,
  use 28,
  split,
  { intros socks_drawn h_length,
    sorry },
  refl,
end

end min_socks_to_guarantee_pairs_l411_411649


namespace find_value_l411_411707

noncomputable def log_add (a b: ℝ) : ℝ :=
  Real.log2 (Real.pow 2 a + Real.pow 2 b)

noncomputable def calc_A : ℝ :=
  log_add (log_add 1 3) 5

noncomputable def calc_B : ℝ :=
  log_add (log_add 2 4) 6

theorem find_value :
  log_add 1 (log_add calc_A calc_B) = 7 := 
sorry

end find_value_l411_411707


namespace proposition_correctness_l411_411337

theorem proposition_correctness :
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬ (1 / a < 1 / b)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ a * b / (a + b)) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (Real.log 9 * Real.log 11 < 1) ∧
  (∀ a b : ℝ, a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → ¬(x + 2 * y = 6)) :=
sorry

end proposition_correctness_l411_411337


namespace necessary_and_sufficient_condition_union_min_formula_value_l411_411181

-- Define sets A and B
variable {α : Type} (A B : Set α)

-- A condition for option B
theorem necessary_and_sufficient_condition_union
  (h : A ∪ B = B) : A ∩ B = A :=
by
  sorry

-- Conditions for option D
variable (x y : ℝ)

-- Assuming x > 1 and y > 1 and x + y = xy
def conditions := x > 1 ∧ y > 1 ∧ x + y = x * y

-- Goal for option D
theorem min_formula_value 
  (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  ∃ (c : ℝ), (min (2 * x / (x - 1) + 4 * y / (y - 1)) = 6 + 4 * real.sqrt 2) :=
by
  sorry

end necessary_and_sufficient_condition_union_min_formula_value_l411_411181


namespace fraction_half_l411_411210

theorem fraction_half {A : ℕ} (h : 8 * (A + 8) - 8 * (A - 8) = 128) (age_eq : A = 64) :
  (64 : ℚ) / (128 : ℚ) = 1 / 2 :=
by
  sorry

end fraction_half_l411_411210


namespace rationalize_denominator_l411_411892

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411892


namespace sphere_radius_unique_l411_411874

noncomputable def cone_radius1 := 1
noncomputable def cone_radius2 := 4
noncomputable def cone_radius3 := 4

noncomputable def apex_angle1 := -4 * Real.arctan (1 / 3)
noncomputable def apex_angle2 := 4 * Real.arctan (9 / 11)
noncomputable def apex_angle3 := 4 * Real.arctan (9 / 11)

theorem sphere_radius_unique
  (r1 r2 r3 : ℝ)
  (α β : ℝ) :
  r1 = cone_radius1 → r2 = cone_radius2 → r3 = cone_radius3 →
  α = apex_angle1 → β = apex_angle2 →
  ∃ (R : ℝ), R = 5 / 3 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  use 5 / 3
  sorry

end sphere_radius_unique_l411_411874


namespace probability_exactly_two_ones_l411_411560

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411560


namespace frequency_sum_l411_411322

theorem frequency_sum (sample_size : ℕ)
  (freq_10_20 freq_20_30 freq_30_40 freq_40_50 freq_50_60 freq_60_70 : ℕ)
  (h_sample_size : sample_size = 20)
  (h_freq_10_20 : freq_10_20 = 2)
  (h_freq_20_30 : freq_20_30 = 3)
  (h_freq_30_40 : freq_30_40 = 4)
  (h_freq_40_50 : freq_40_50 = 5)
  (h_freq_50_60 : freq_50_60 = 4)
  (h_freq_60_70 : freq_60_70 = 2) :
  freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50 = 14 :=
by {
  rw [h_freq_10_20, h_freq_20_30, h_freq_30_40, h_freq_40_50],
  norm_num,
  sorry
}

end frequency_sum_l411_411322


namespace soccer_red_cards_l411_411021

theorem soccer_red_cards (total_players yellow_carded_players : ℕ)
  (no_caution_players : total_players = 11)
  (five_no_cautions : 5 = 5)
  (players_received_yellow : yellow_carded_players = total_players - 5)
  (yellow_per_player : ∀ p, p = 6 -> 6 * 1)
  (red_card_rule : ∀ y, (yellow_carded_players * 1) = y -> y / 2 = 3) :
  ∃ red_cards : ℕ, red_cards = 3 :=
by { 
  existsi 3,
  sorry
}

end soccer_red_cards_l411_411021


namespace angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l411_411061

-- Definitions according to the given conditions
variables (A B C D E F M N P : Point)
  (hexagon_regular : is_regular_hexagon A B C D E F)
  (is_midpoint_M : is_midpoint M C D)
  (is_midpoint_N : is_midpoint N D E)
  (intersection_P : intersection_point P (line_through A M) (line_through B N))

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_60_degrees 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  angle (line_through A M) (line_through B N) = 60 := 
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_triangle_ABP_eq_area_quadrilateral_MDNP 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  area (triangle A B P) = area (quadrilateral M D N P) := 
sorry

end angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l411_411061


namespace probability_two_ones_in_twelve_dice_l411_411589
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411589


namespace area_of_triangle_from_tangent_line_l411_411914

noncomputable def tangent_line_at_point (x : ℝ) : ℝ := 2 * x + 2

def line_y_equals_x (x : ℝ) : ℝ := x

def line_y_equals_zero (x : ℝ) : ℝ := 0

def intersection_with_x_axis : ℝ := -1

def intersection_with_y_equals_x : ℝ := -2

def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_from_tangent_line :
  let base := real.abs ((intersection_with_y_equals_x - 0) - (intersection_with_x_axis - 0)),
      height := real.abs (intersection_with_y_equals_x - (line_y_equals_x intersection_with_y_equals_x))
  in area_of_triangle base height = 3 := by
  sorry

end area_of_triangle_from_tangent_line_l411_411914


namespace number_of_values_n_l411_411507

-- Defining the given polynomial with integer coefficients
def polynomial (x : ℝ) : ℝ := x^3 - 3003 * x^2 + (17 * (1332^2) / 16 - r^2) * x - 1332 * ((1332^2) / 16 - r^2)

-- Conditions in the problem
def integer_zero := 1332
def irrational_prod (r_sq : ℝ) := ∀ r : ℝ, r^2 = r_sq -> ¬ (∃ (n : ℤ), r = n)
def distinct_positives (r_sq : ℝ) := 0 < r_sq ∧ r_sq < (integer_zero^2 / 16)

-- The main theorem statement
theorem number_of_values_n : ∃ (r_sq_set : set ℝ), (∀ r_sq ∈ r_sq_set, irrational_prod r_sq ∧ distinct_positives r_sq) ∧ r_sq_set.card = 110888 :=
by
  sorry

end number_of_values_n_l411_411507


namespace ab_value_eq_10_l411_411759

theorem ab_value_eq_10 :
  let x1 := Real.cbrt (17 - (27 / 4) * Real.sqrt 6)
  let x2 := Real.cbrt (17 + (27 / 4) * Real.sqrt 6)
  let a := x1 + x2
  let b := x1 * x2
  (a * b) = 10 :=
by
  let x1 := Real.cbrt (17 - (27 / 4) * Real.sqrt 6)
  let x2 := Real.cbrt (17 + (27 / 4) * Real.sqrt 6)
  let a := x1 + x2
  let b := x1 * x2
  sorry

end ab_value_eq_10_l411_411759


namespace quoted_price_of_shares_l411_411212

theorem quoted_price_of_shares (investment : ℝ) (face_value : ℝ) (rate_dividend : ℝ) (annual_income : ℝ) (num_shares : ℝ) (quoted_price : ℝ) :
  investment = 4455 ∧ face_value = 10 ∧ rate_dividend = 0.12 ∧ annual_income = 648 ∧ num_shares = annual_income / (rate_dividend * face_value) →
  quoted_price = investment / num_shares :=
by sorry

end quoted_price_of_shares_l411_411212


namespace no_closed_path_in_2019_square_l411_411820

theorem no_closed_path_in_2019_square : 
  ∀ (square : ℕ → ℕ → Prop), 
    (∀ x y, square x y ↔ (0 <= x ∧ x < 2019) ∧ (0 <= y ∧ y < 2019)) →
    ¬ ∃ path : list (ℕ × ℕ), 
      (∀ (i : ℕ), i < path.length - 1 → 
        (let (x1, y1) := path.nth_le i sorry in
         let (x2, y2) := path.nth_le (i + 1) sorry in
         (x1 = x2 ∨ y1 = y2) ∧ (|x1 - x2| ≤ 1 ∧ |y1 - y2| ≤ 1))) ∧ 
      (path.head = path.last) ∧ 
      (∀ (x y : ℕ), (0 <= x ∧ x < 2019 ∧ 0 <= y ∧ y < 2019) → 
        (x, y) ∈ path) :=
begin
  sorry
end

end no_closed_path_in_2019_square_l411_411820


namespace quadrilateral_area_l411_411395

def quadrilateral_vertices : list (ℝ × ℝ) := [(4, -3), (4, 7), (12, 2), (12, -7)]

theorem quadrilateral_area (h : quadrilateral_vertices = [(4, -3), (4, 7), (12, 2), (12, -7)]) :
  quadrilateral_area quadrilateral_vertices = 76 :=
sorry

end quadrilateral_area_l411_411395


namespace team_red_cards_l411_411017

-- Define the conditions
def team_size (n : ℕ) := n = 11
def players_without_caution (n : ℕ) := n = 5
def yellow_cards_per_cautioned_player := 1
def yellow_cards_per_red_card := 2

-- Define the proof statement
theorem team_red_cards : 
  ∀ (total_players cautioned_players yellow_cards total_red_cards : ℕ),
  team_size total_players →
  players_without_caution (total_players - cautioned_players) →
  cautioned_players * yellow_cards_per_cautioned_player = yellow_cards →
  yellow_cards / yellow_cards_per_red_card = total_red_cards →
  total_red_cards = 3 :=
by
  intros total_players cautioned_players yellow_cards total_red_cards
  assume h1 h2 h3 h4
  sorry

end team_red_cards_l411_411017


namespace L_shape_count_l411_411691

def L_shape_orientations : List (List (Fin 3 × Fin 3)) :=
  [ [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨1, by simp⟩)],  -- Original orientation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨0, by simp⟩, ⟨1, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩)],  -- 90° rotation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩)],  -- 180° rotation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨0, by simp⟩, ⟨1, by simp⟩), (⟨1, by simp⟩, ⟨1, by simp⟩)] ]-- 270° rotation

def L_shape_in_grid (shape : List (Fin 3 × Fin 3)) : List (Fin 3 × Fin 3) :=
  -- Function to determine if shape is within grid:
  if shape.all (λ (x : Fin 3 × Fin 3), x.1 < 3 ∧ x.2 < 3)
  then shape
  else []

theorem L_shape_count : ∃ n : Nat, n = 48 :=
by { let placements := list.bind L_shape_orientations (λ o, list.map (λ (f : (Fin 3 × Fin 3)), L_shape_in_grid o) [⟨0, by simp⟩, ⟨1, by simp⟩, ⟨2, by simp⟩]),
     have h := placements.length,
     have h48 : h = 48, from sorry,
     exact ⟨h, h48⟩ }

end L_shape_count_l411_411691


namespace sycamore_birch_distance_is_two_l411_411468

noncomputable theory

structure TreeRow :=
  (poplar willow locust birch sycamore : ℤ)
  (dist_1 : willow = poplar + 1 ∨ willow = poplar - 1)
  (dist_2 : locust = poplar + 1 ∨ locust = poplar - 1)
  (dist_3 : poplar_willow_eq_poplar_locust: abs (willow - poplar) = abs (locust - poplar))
  (dist_4 : abs (birch - poplar) = abs (birch - locust))

def sycamore_birch_distance (row : TreeRow) : ℤ :=
  abs (row.sycamore - row.birch)

theorem sycamore_birch_distance_is_two (row : TreeRow) : sycamore_birch_distance row = 2 :=
by
  sorry

end sycamore_birch_distance_is_two_l411_411468


namespace solution_length_of_table_l411_411465

noncomputable def length_of_table (occupied_width : ℕ) (initial_width : ℕ) (sheet_height : ℕ) (sheet_width : ℕ) : ℕ :=
  let total_sheets := occupied_width - sheet_width in
  sheet_height + total_sheets

def length_of_table_problem :=
  let table_width := 80
  let initial_sheet_height := 5
  let initial_sheet_width := 8
  length_of_table table_width initial_sheet_width initial_sheet_height initial_sheet_width = 77

theorem solution_length_of_table : length_of_table_problem :=
  by
    sorry

end solution_length_of_table_l411_411465


namespace newcomen_first_steam_engine_patent_l411_411409

theorem newcomen_first_steam_engine_patent :
  (∃ (t : ℕ), t = 1712) ∧ (∃ y : ℕ, y = 1705) ∧
    (∀ p : string, p = "Thomas Newcomen" → (∃ e, e = "steam engine patent")) :=
by sorry

end newcomen_first_steam_engine_patent_l411_411409


namespace ratio_Rose_to_Mother_l411_411881

variable (Rose_age : ℕ) (Mother_age : ℕ)

-- Define the conditions
axiom sum_of_ages : Rose_age + Mother_age = 100
axiom Rose_is_25 : Rose_age = 25
axiom Mother_is_75 : Mother_age = 75

-- Define the main theorem to prove the ratio
theorem ratio_Rose_to_Mother : (Rose_age : ℚ) / (Mother_age : ℚ) = 1 / 3 := by
  sorry

end ratio_Rose_to_Mother_l411_411881


namespace probability_given_range_l411_411336

theorem probability_given_range (a : ℝ) : 
  (∀ k : ℕ, k ≥ 2 → P(ξ = k) = 1 / 2^(k-1)) → P(ξ = 1) = a → P(2 < ξ ≤ 5) = 7 / 16 := 
by
  sorry

end probability_given_range_l411_411336


namespace find_c_value_l411_411985

theorem find_c_value :
  (∀ x : ℝ, (x * (2 * x + 1) < c ↔ x ∈ Ioo (-2 : ℝ) (3 / 2))) → c = -3 :=
begin
  intro h,
  sorry
end

end find_c_value_l411_411985


namespace determine_a_l411_411282

theorem determine_a (a : ℝ) :
  (∃ (x y : ℝ), (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0 ∧ (x + 3)^2 + (y - 5)^2 = a) →
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end determine_a_l411_411282


namespace zog_word_count_l411_411875

noncomputable def num_zoggian_words (alphabet_size : ℕ) (max_length : ℕ) : ℕ :=
  (Σ i in finset.range (max_length + 1), if i = 0 then 0 else alphabet_size ^ i)

theorem zog_word_count : num_zoggian_words 6 4 = 1554 :=
  by
    -- we can calculate it manually here or use the property directly
    -- Σ i ∈ {0, 1, 2, 3, 4}, if i = 0 then 0 else 6 ^ i
    have h : (finset.range 5).sum (λ i, if i = 0 then 0 else 6 ^ i) = 6 + 36 + 216 + 1296 := by simp,
    rw h,
    norm_num,
    sorry -- skips the detailed proof steps

end zog_word_count_l411_411875


namespace find_r_l411_411357

noncomputable theory

def vecA : ℝ × ℝ × ℝ := (2, 3, -1)
def vecB : ℝ × ℝ × ℝ := (1, 1, 0)

def crossProduct (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def vecC : ℝ × ℝ × ℝ := (3, 4, -1)

def scalarEquation (p q r : ℝ) : Prop :=
  (vecC.1 = p * vecA.1 + q * vecB.1 + r * (crossProduct vecA vecB).1) ∧
  (vecC.2 = p * vecA.2 + q * vecB.2 + r * (crossProduct vecA vecB).2) ∧
  (vecC.3 = p * vecA.3 + q * vecB.3 + r * (crossProduct vecA vecB).3)

theorem find_r : ∃ r : ℝ, scalarEquation 0 0 r ∧ r = 2 / 3 :=
by
  sorry

end find_r_l411_411357


namespace pentagon_area_is_120_l411_411831

noncomputable def BE (AE : ℝ) : ℝ := AE / Real.sqrt 2
noncomputable def CE (BE : ℝ) : ℝ := BE / Real.sqrt 2
noncomputable def DE (CE : ℝ) : ℝ := CE / Real.sqrt 2
noncomputable def FE (DE : ℝ) : ℝ := DE / Real.sqrt 2

noncomputable def Area (AE BE CE DE FE : ℝ) : ℝ :=
  (1 / 2) * BE * BE + (1 / 2) * CE * CE + (1 / 2) * DE * DE + (1 / 2) * FE * FE

theorem pentagon_area_is_120 (AE : ℝ) (h AE = 16) :
  Area AE (BE AE) (CE (BE AE)) (DE (CE (BE AE))) (FE (DE (CE (BE AE)))) = 120 := by
  sorry

end pentagon_area_is_120_l411_411831


namespace cannot_tile_regular_pentagon_l411_411217

theorem cannot_tile_regular_pentagon :
  ¬ (∃ n : ℕ, 360 % (180 - (360 / 5 : ℕ)) = 0) :=
by sorry

end cannot_tile_regular_pentagon_l411_411217


namespace correct_propositions_in_regression_analysis_l411_411239

theorem correct_propositions_in_regression_analysis :
  let p1 := "In a linear regression model, \overset{\land }{e} represents the random error of the predicted value \overset{\land }{b}x + \overset{\land }{a} from the actual value y, and it is an observable quantity."
  let p2 := "The smaller the sum of squared residuals of a model, the better the fitting effect."
  let p3 := "Using R^2 to describe the regression equation, the smaller the R^2, the better the fitting effect."
  let p4 := "In the residual plot, if the residual points are evenly distributed in a horizontal band area, it indicates that the chosen model is appropriate. If the band area is narrower, it indicates a higher fitting precision and a higher prediction accuracy of the regression equation."
  let option_C := [p1, p4]
in option_C = ["In a linear regression model, \overset{\land }{e} represents the random error of the predicted value \overset{\land }{b}x + \overset{\land }{a} from the actual value y, and it is an observable quantity.",
                "In the residual plot, if the residual points are evenly distributed in a horizontal band area, it indicates that the chosen model is appropriate. If the band area is narrower, it indicates a higher fitting precision and a higher prediction accuracy of the regression equation."] := sorry

end correct_propositions_in_regression_analysis_l411_411239


namespace exists_constant_for_inequality_l411_411878

noncomputable def H (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), 1 / (k + 1 : ℝ)

theorem exists_constant_for_inequality :
  ∃ C > 0, ∀ (m : ℕ) (a : Fin m → ℕ), 
    (∑ i in Finset.range m, H (a i)) ≤ C * (∑ i in Finset.range m, i.succ * (a i)) ^ (1 / 2 : ℝ) :=
sorry

end exists_constant_for_inequality_l411_411878


namespace red_cards_needed_l411_411014

-- Define the initial conditions
def total_players : ℕ := 11
def players_without_cautions : ℕ := 5
def yellow_cards_per_player : ℕ := 1
def yellow_cards_per_red_card : ℕ := 2

-- Theorem statement for the problem
theorem red_cards_needed (total_players = 11) (players_without_cautions = 5) 
    (yellow_cards_per_player = 1) (yellow_cards_per_red_card = 2) : (total_players - players_without_cautions) * yellow_cards_per_player / yellow_cards_per_red_card = 3 := 
by
  sorry

end red_cards_needed_l411_411014


namespace solve_ellipseProblem_l411_411270

noncomputable def ellipseProblem (m : ℝ) : Prop :=
  (m > 0) ∧ (∃ x y : ℝ, (x^2 / (m^2 + 1) + y^2 / m^2 = 1) ∧ 
             (let F1 : ℝ × ℝ := (-1, 0); F2 : ℝ × ℝ := (1, 0); 
                  P : ℝ × ℝ := (0, m) in
              ∠ P (↑(x,y)) F1 F2 = 2 * Real.pi / 3))

theorem solve_ellipseProblem (m : ℝ) (h : ellipseProblem m) : m = Real.sqrt(3) / 3 :=
  sorry

end solve_ellipseProblem_l411_411270


namespace square_area_l411_411942

theorem square_area (side_length : ℕ) (h : side_length = 11) : side_length * side_length = 121 :=
by
  rw h
  norm_num
  sorry

end square_area_l411_411942


namespace option_d_correct_l411_411989

theorem option_d_correct (m n : ℝ) : (m + n) * (m - 2 * n) = m^2 - m * n - 2 * n^2 :=
by
  sorry

end option_d_correct_l411_411989


namespace train_route_l411_411467

-- Definition of letter positions
def letter_position : Char → Nat
| 'A' => 1
| 'B' => 2
| 'K' => 11
| 'L' => 12
| 'U' => 21
| 'V' => 22
| _ => 0

-- Definition of decode function
def decode (s : List Nat) : String :=
match s with
| [21, 2, 12, 21] => "Baku"
| [21, 22, 12, 21] => "Ufa"
| _ => ""

-- Assert encoded strings
def departure_encoded : List Nat := [21, 2, 12, 21]
def arrival_encoded : List Nat := [21, 22, 12, 21]

-- Theorem statement
theorem train_route :
  decode departure_encoded = "Ufa" ∧ decode arrival_encoded = "Baku" :=
by
  sorry

end train_route_l411_411467


namespace area_of_third_polygon_l411_411529

theorem area_of_third_polygon (S₁ S₂ : ℝ) : 
  ∃ S₃ : ℝ, S₃ = sqrt (2 * S₂^3 / (S₁ + S₂)) :=
by
  use sqrt (2 * S₂^3 / (S₁ + S₂))
  sorry

end area_of_third_polygon_l411_411529


namespace sum_of_coefficients_l411_411177

/-- Given the expansion of (5x + 3y + 2)(2x + 5y + 7), 
    prove that the sum of the coefficients of the terms 
    that contain a nonzero power of y is equal to 77. -/
theorem sum_of_coefficients :
  let expr := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 7)
  let expanded := expand expression
  let terms_with_y := coeffs_containing y expanded
  coeffs_sum terms_with_y = 77 := sorry

end sum_of_coefficients_l411_411177


namespace circle_equation_tangent_l411_411724

theorem circle_equation_tangent (h : ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = 25)) :
    ∃ c : ℝ × ℝ, c = (1, 2) ∧ ∃ r : ℝ, r = 5 ∧ ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) := 
by
    sorry

end circle_equation_tangent_l411_411724


namespace unit_price_first_purchase_l411_411693

theorem unit_price_first_purchase (x y : ℝ) (h1 : x * y = 500000) 
    (h2 : 1.4 * x * (y + 10000) = 770000) : x = 5 :=
by
  -- Proof details here
  sorry

end unit_price_first_purchase_l411_411693


namespace total_marbles_l411_411007

theorem total_marbles (ratio_red_blue_green_yellow : ℕ → ℕ → ℕ → ℕ → Prop) (total : ℕ) :
  (∀ r b g y, ratio_red_blue_green_yellow r b g y ↔ r = 1 ∧ b = 5 ∧ g = 3 ∧ y = 2) →
  (∃ y, y = 20) →
  (total = y * 11 / 2) →
  total = 110 :=
by
  intros ratio_condition yellow_condition total_condition
  sorry

end total_marbles_l411_411007


namespace distance_from_M_to_directrix_l411_411349

theorem distance_from_M_to_directrix (p : ℝ) (hp : p > 0) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x → y) 
  (line_intersects_parabola : ∀ (A B : ℝ × ℝ), A.snd = B.snd → ∃ (m : ℝ × ℝ), m.snd = 2) : 
  ∃ (distance : ℝ), distance = 4 :=
by
  sorry

end distance_from_M_to_directrix_l411_411349


namespace max_abs_w_l411_411436

theorem max_abs_w (p q r w : ℂ) (s : ℝ) (hs : s > 0) 
(hp : |p| = s) (hq : |q| = s) (hr : |r| = s)
(h : p * w^3 + q * w^2 + r * w + r = 0) :
  |w| ≤ 1.4656 :=
sorry

end max_abs_w_l411_411436


namespace calories_per_strawberry_l411_411188

theorem calories_per_strawberry (x : ℕ) :
  (12 * x + 6 * 17 = 150) → x = 4 := by
  sorry

end calories_per_strawberry_l411_411188


namespace sugar_percentage_first_solution_l411_411079

theorem sugar_percentage_first_solution 
  (x : ℝ) (h1 : 0 < x ∧ x < 100) 
  (h2 : 17 = 3 / 4 * x + 1 / 4 * 38) : 
  x = 10 :=
sorry

end sugar_percentage_first_solution_l411_411079


namespace solve_z_cubic_l411_411725

def z_solutions (x y : ℝ) : ℂ :=
  x + y * complex.I

theorem solve_z_cubic :
  ∃ x y : ℝ, z_solutions x y ^ 6 + 6 * complex.I = 0 :=
sorry

end solve_z_cubic_l411_411725


namespace axis_of_parabola_l411_411723

-- Define the given equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8 * y

-- Define the standard form of a vertical parabola and the value we need to prove (axis of the parabola)
def standard_form (p y : ℝ) : Prop := y = 2

-- The proof problem: Given the equation of the parabola, prove the equation of its axis.
theorem axis_of_parabola : 
  ∀ x y : ℝ, (parabola x y) → (standard_form y 2) :=
by
  intros x y h
  sorry

end axis_of_parabola_l411_411723


namespace surface_area_increase_156_percent_l411_411176

variable {s : ℝ} (h : s > 0)

def original_surface_area : ℝ := 6 * s^2
def new_surface_area : ℝ := 6 * (1.6 * s)^2
def increase_in_surface_area : ℝ := new_surface_area - original_surface_area
def percent_increase : ℝ := (increase_in_surface_area / original_surface_area) * 100

theorem surface_area_increase_156_percent : 
  percent_increase = 156 := by
  unfold original_surface_area new_surface_area increase_in_surface_area percent_increase
  sorry

end surface_area_increase_156_percent_l411_411176


namespace find_O1O2_l411_411610

def radius_1 := 4
def radius_2 := 3
def ratio := 2 / Real.sqrt 3 

-- Given conditions
structure Problem :=
  (radius1 : ℝ)
  (radius2 : ℝ)
  (O1O2_ratio_M1M2 : ℝ)

-- Create an instance of Problem with given conditions
def givenProblem : Problem :=
  { radius1 := radius_1,
    radius2 := radius_2,
    O1O2_ratio_M1M2 := ratio }

-- Problem statement to prove that the length O1O2 = 14
theorem find_O1O2 (prob : Problem)
  (h1 : prob.radius1 = 4)
  (h2 : prob.radius2 = 3)
  (h3 : prob.O1O2_ratio_M1M2 = 2 / Real.sqrt 3) :
  let O1O2 : ℝ := 14 in
  O1O2 = 14 :=
by
  sorry

end find_O1O2_l411_411610


namespace max_sum_sqrt_l411_411708

theorem max_sum_sqrt (x : ℕ → ℝ) (hx : ∀ i < 4, 0 ≤ x i) (h_sum : ∑ i in finset.range 4, x i = 1) :
  (∑ i in finset.range 4, ∑ j in (finset.range 4).filter (λ j, i < j), (x i + x j) * real.sqrt(x i * x j)) ≤ 3 / 4 ∧ 
  (∃ v : ℕ, v < 4 ∧ x v = 1 / 4) := 
sorry

end max_sum_sqrt_l411_411708


namespace words_memorized_l411_411097

theorem words_memorized (x y z : ℕ) (h1 : x = 4 * (y + z) / 5) (h2 : x + y = 6 * z / 5) (h3 : 100 < x + y + z ∧ x + y + z < 200) : 
  x + y + z = 198 :=
by
  sorry

end words_memorized_l411_411097


namespace probability_two_dice_showing_1_l411_411603

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411603


namespace total_oranges_picked_is_correct_l411_411870

constant orangeTree1 : ℕ := 80
constant orangeTree2 : ℕ := 60
constant orangeTree3 : ℕ := 120
constant orangeTree4 : ℕ := 45
constant orangeTree5 : ℕ := 25
constant orangeTree6 : ℕ := 97

theorem total_oranges_picked_is_correct : 
  orangeTree1 + orangeTree2 + orangeTree3 + orangeTree4 + orangeTree5 + orangeTree6 = 427 := 
  sorry

end total_oranges_picked_is_correct_l411_411870


namespace probability_two_dice_showing_1_l411_411604

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411604


namespace swimming_pool_distance_l411_411420

theorem swimming_pool_distance (julien_daily_distance : ℕ) (sarah_multi_factor : ℕ)
    (jamir_additional_distance : ℕ) (week_days : ℕ) 
    (julien_weekly_distance : ℕ) (sarah_weekly_distance : ℕ) (jamir_weekly_distance : ℕ) 
    (total_combined_distance : ℕ) : 
    julien_daily_distance = 50 → 
    sarah_multi_factor = 2 →
    jamir_additional_distance = 20 →
    week_days = 7 →
    julien_weekly_distance = julien_daily_distance * week_days →
    sarah_weekly_distance = (sarah_multi_factor * julien_daily_distance) * week_days →
    jamir_weekly_distance = ((sarah_multi_factor * julien_daily_distance) + jamir_additional_distance) * week_days →
    total_combined_distance = julien_weekly_distance + sarah_weekly_distance + jamir_weekly_distance →
    total_combined_distance = 1890 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4] at *
  rw [h5, h6, h7, h8]
  sorry

end swimming_pool_distance_l411_411420


namespace dihedral_angle_of_equilateral_triangle_l411_411398

theorem dihedral_angle_of_equilateral_triangle (a : ℝ) 
(ABC_eq : ∀ {A B C : ℝ}, (B - A) ^ 2 + (C - A) ^ 2 = a^2 ∧ (C - B) ^ 2 + (A - B) ^ 2 = a^2 ∧ (A - C) ^ 2 + (B - C) ^ 2 = a^2) 
(perpendicular : ∀ A B C D : ℝ, D = (B + C)/2 ∧ (B - D) * (C - D) = 0) : 
∃ θ : ℝ, θ = 60 := 
  sorry

end dihedral_angle_of_equilateral_triangle_l411_411398


namespace inequality_solution_l411_411933

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x - 3) ≥ 1 ↔ (x > 3 ∨ x ≤ -2) :=
by 
  sorry

end inequality_solution_l411_411933


namespace f_injective_l411_411847

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def inS (n : ℕ) : Prop :=
  n > 0 ∧ ¬ isPerfectSquare n

def min_ar (n : ℕ) (a_r : ℕ) : Prop :=
  ∃ (a : List ℕ), a.last? = some a_r ∧ n < a.head ∧ a.pairwise (≤) ∧ isPerfectSquare (n * a.prod)

def f (n : ℕ) : ℕ :=
  if h : ∃ a_r : ℕ, min_ar n a_r then Classical.choose h else n

theorem f_injective : ∀ n m : ℕ, inS n → inS m → f n = f m → n = m := 
  by
  intros n m hn hm hnm
  sorry

end f_injective_l411_411847


namespace mary_puts_back_correct_number_of_oranges_l411_411633

namespace FruitProblem

def price_apple := 40
def price_orange := 60
def total_fruits := 10
def average_price_all := 56
def average_price_kept := 50

theorem mary_puts_back_correct_number_of_oranges :
  ∀ (A O O' T: ℕ),
  A + O = total_fruits →
  A * price_apple + O * price_orange = total_fruits * average_price_all →
  A = 2 →
  T = A + O' →
  A * price_apple + O' * price_orange = T * average_price_kept →
  O - O' = 6 :=
by
  sorry

end FruitProblem

end mary_puts_back_correct_number_of_oranges_l411_411633


namespace sum_factorials_last_two_digits_l411_411970

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l411_411970


namespace friendship_zero_g_is_friendship_fixed_point_friendship_l411_411640

-- Define the predicate for a Friendship Function
def isFriendshipFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ set.Icc 0 1, 0 ≤ f x) ∧
  (f 1 = 1) ∧
  (∀ x1 x2 ∈ set.Icc 0 1, x1 + x2 ∈ set.Icc 0 1 → f (x1 + x2) ≥ f x1 + f x2)

-- Question 1: Prove that f(0) = 0 for Friendship Functions
theorem friendship_zero (f : ℝ → ℝ) (h : isFriendshipFunction f) : f 0 = 0 :=
by sorry

-- Question 2: Prove that the function g(x) = 2x - 1 is a Friendship Function
def g (x : ℝ) := 2 * x - 1

theorem g_is_friendship : isFriendshipFunction g :=
by sorry

-- Question 3: Prove that if f is a Friendship Function, and there exists x0 such that f(x0) ∈ [0,1] and f(f(x0)) = x0, then f(x0) = x0
theorem fixed_point_friendship (f : ℝ → ℝ) (h : isFriendshipFunction f) (x0 : ℝ) 
  (hx0 : x0 ∈ set.Icc 0 1) (hfx0 : f x0 ∈ set.Icc 0 1) (hffx0 : f (f x0) = x0) : f x0 = x0 :=
by sorry

end friendship_zero_g_is_friendship_fixed_point_friendship_l411_411640


namespace last_two_digits_of_sum_l411_411956

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l411_411956


namespace age_ratio_proof_l411_411997

-- Definitions according to the conditions
def P : ℕ := 33 -- p's current age, since p was 30 years old 3 years ago
def Q : ℕ := (44 / 2) -- we solve for q's current age from P + 11 = 2(Q + 11)

-- Lean 4 statement to prove
theorem age_ratio_proof : P / Q = 3 := by
  have h1 : P + 11 = 2*(Q + 11) := by sorry
  have h2 : 2*Q = 22 := by sorry
  have h3 : Q = 11 := by sorry
  rw [h3]
  show P / 11 = 3
  calc P / 11 = 33 / 11 : by sorry
           ... = 3      : by sorry

end age_ratio_proof_l411_411997


namespace yogurt_cost_l411_411157

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l411_411157


namespace control_plant_height_l411_411044

-- Given Conditions
variable (H : ℝ)
variable (bone_meal_height : ℝ := 1.25 * H)
variable (cow_manure_height : ℝ := 2 * bone_meal_height)
variable (cow_manure_height_value : ℝ := 90)

-- Goal
theorem control_plant_height : H = 36 :=
by
  -- Definitions and given conditions
  have h1 : bone_meal_height = 1.25 * H := rfl
  have h2 : cow_manure_height = 2 * bone_meal_height := rfl
  have h3 : cow_manure_height = 90 := rfl
  sorry

end control_plant_height_l411_411044


namespace contacts_per_dollar_l411_411679

theorem contacts_per_dollar :
  (let cost_per_contact_first := 50 / 25 in
   let cost_per_contact_second := 99 / 33 in
   cost_per_contact_second > cost_per_contact_first → 3 = 3) := 
begin
  sorry
end

end contacts_per_dollar_l411_411679


namespace ratio_lateral_surface_areas_l411_411669

variables (α β : ℝ)

theorem ratio_lateral_surface_areas :
  let S1 := 2 * sqrt(2) * cot β,
      S2 := (2 * sin (α + β) * sqrt(2 * (1 + sin α ^ 2))) / (2 * sin α ^ 2 * cos β)
  in S2 / S1 = (sin (α + β) * sqrt(2 * (1 + sin α ^ 2))) / (2 * sin α ^ 2 * cos β) := 
by sorry

end ratio_lateral_surface_areas_l411_411669


namespace alpha_eq_pi_over_3_l411_411738

theorem alpha_eq_pi_over_3 (α β γ : ℝ) (h1 : 0 < α ∧ α < π) (h2 : α + β + γ = π) 
    (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
    α = π / 3 :=
by
  sorry

end alpha_eq_pi_over_3_l411_411738


namespace green_papayas_left_l411_411144

/-- Define the initial number of green papayas on the tree -/
def initial_green_papayas : ℕ := 14

/-- Define the number of papayas that turned yellow on Friday -/
def friday_yellow_papayas : ℕ := 2

/-- Define the number of papayas that turned yellow on Sunday -/
def sunday_yellow_papayas : ℕ := 2 * friday_yellow_papayas

/-- The remaining number of green papayas after Friday and Sunday -/
def remaining_green_papayas : ℕ := initial_green_papayas - friday_yellow_papayas - sunday_yellow_papayas

theorem green_papayas_left : remaining_green_papayas = 8 := by
  sorry

end green_papayas_left_l411_411144


namespace fernandez_family_children_l411_411481

-- Conditions definition
variables (m : ℕ) -- age of the mother
variables (x : ℕ) -- number of children
variables (y : ℕ) -- average age of the children

-- Given conditions
def average_age_family (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + 50 + 70 + x * y) / (3 + x) = 25

def average_age_mother_children (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + x * y) / (1 + x) = 18

-- Goal statement
theorem fernandez_family_children
  (m : ℕ) (x : ℕ) (y : ℕ)
  (h1 : average_age_family m x y)
  (h2 : average_age_mother_children m x y) :
  x = 9 :=
sorry

end fernandez_family_children_l411_411481


namespace find_n_l411_411829

open Nat

def a (n : ℕ) : ℕ
| 0     => 2
| n + 1 => 2 * a n

def S (n : ℕ) : ℕ := (range (n+1)).sum a

theorem find_n (n : ℕ) (h : S n = 126) : n = 6 :=
sorry

end find_n_l411_411829


namespace maximum_n_Sn_pos_l411_411326

def arithmetic_sequence := ℕ → ℝ

noncomputable def sum_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

axiom a1_eq : ∀ (a : arithmetic_sequence), (a 1) = 2 * (a 2) + (a 4)

axiom S5_eq_5 : ∀ (a : arithmetic_sequence), sum_first_n_terms a 5 = 5

theorem maximum_n_Sn_pos : ∀ (a : arithmetic_sequence), (∃ (n : ℕ), n < 6 ∧ sum_first_n_terms a n > 0) → n = 5 :=
  sorry

end maximum_n_Sn_pos_l411_411326


namespace max_african_team_wins_max_l411_411010

-- Assume there are n African teams and (n + 9) European teams.
-- Each pair of teams plays exactly once.
-- European teams won nine times as many matches as African teams.
-- Prove that the maximum number of matches that a single African team might have won is 11.

theorem max_african_team_wins_max (n : ℕ) (k : ℕ) (n_african_wins : ℕ) (n_european_wins : ℕ)
  (h1 : n_african_wins = (n * (n - 1)) / 2) 
  (h2 : n_european_wins = ((n + 9) * (n + 8)) / 2 + k)
  (h3 : n_european_wins = 9 * (n_african_wins + (n * (n + 9) - k))) :
  ∃ max_wins, max_wins = 11 := by
  sorry

end max_african_team_wins_max_l411_411010


namespace inequality_sum_ratios_squares_l411_411307

theorem inequality_sum_ratios_squares (n : ℕ) (a : Finₙ → ℝ) (h1: ∀ i, 0 < a i) (h2: ∏ i, a i = 1) :
  ∑ i, (a i / a ((i + 1) % n)) ^ (n - 1) ≥ ∑ i, (a i) ^ 2 :=
sorry

end inequality_sum_ratios_squares_l411_411307


namespace part1_part2_l411_411451

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

theorem part1 (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : x > 0) : f a x ≤ -1 :=
sorry

theorem part2 (a : ℝ) (θ : ℝ) (h1 : a ≥ 1) (h2 : 0 ≤ θ) (h3 : θ ≤ Real.pi / 2) : 
  f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ) :=
sorry

end part1_part2_l411_411451


namespace second_platform_length_is_250_l411_411226

def train := 150 -- length of the train in meters
def platform1 := 150 -- length of the first platform in meters
def time1 := 15 -- time to cross the first platform in seconds
def time2 := 20 -- time to cross the second platform in seconds

noncomputable def speed := (train + platform1) / time1 -- speed of the train in m/sec

def second_platform_length : ℝ := (speed * time2) - train

theorem second_platform_length_is_250 : second_platform_length = 250 :=
by
  -- We calculate the second platform length based on conditions and prove it equals 250
  sorry

end second_platform_length_is_250_l411_411226


namespace water_volume_into_sea_per_minute_l411_411666

noncomputable def volume_per_minute := 
  let depth_A := 5
  let width_A := 35
  let flow_rate_A := 2 * 1000 / 60
  let depth_B := 7
  let width_B := 45
  let flow_rate_B := 3 * 1000 / 60

  let average_depth := (depth_A + depth_B) / 2
  let average_width := (width_A + width_B) / 2
  let average_area := average_depth * average_width

  let average_flow_rate := (flow_rate_A + flow_rate_B) / 2

  average_area * average_flow_rate

theorem water_volume_into_sea_per_minute : volume_per_minute ≈ 10000.8 := 
  sorry

end water_volume_into_sea_per_minute_l411_411666


namespace complement_of_M_in_U_l411_411453

open Set

-- Definition: Universal set and subset
def U : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}

-- Theorem statement
theorem complement_of_M_in_U :
  compl U M = {-3, -4} := 
by
  sorry

end complement_of_M_in_U_l411_411453


namespace product_gcd_lcm_l411_411296

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l411_411296


namespace kelly_initial_apples_l411_411425

theorem kelly_initial_apples : ∀ (T P I : ℕ), T = 105 → P = 49 → I + P = T → I = 56 :=
by
  intros T P I ht hp h
  rw [ht, hp] at h
  linarith

end kelly_initial_apples_l411_411425


namespace F_at_2_l411_411694

def F (x : ℝ) : ℝ :=
  sqrt (abs (x - 1)) + (10 / Real.pi) * Real.arctan (sqrt (abs x))

theorem F_at_2 : F 2 = 4 :=
  sorry

end F_at_2_l411_411694


namespace num_divisors_exponent_1728_l411_411370

theorem num_divisors_exponent_1728 {n : ℕ} :
  let base := 1728,
      exponent := 1728 in
  n = base ^ exponent →
  let divisors := (10368 + 1) * (5184 + 1) in
  let condition := 1728 in
  (∃ d ∈ (nat.divisors n), (nat.divisors d).card = condition) →
    20 :=
begin
  sorry
end

end num_divisors_exponent_1728_l411_411370


namespace general_formula_a_sum_T_max_k_value_l411_411324

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end general_formula_a_sum_T_max_k_value_l411_411324


namespace initial_girls_count_l411_411209

theorem initial_girls_count 
  (p : ℝ) 
  (h₁ : 0.6 * p - 3 = 0.5 * p)
  : 0.6 * p = 18 :=
by {
  have h₂ : 0.1 * p = 3 := by linarith,
  have h₃ : p = 30 := by linarith,
  show 0.6 * p = 18, by linarith,
  sorry
}

end initial_girls_count_l411_411209


namespace solve_for_D_d_Q_R_l411_411373

theorem solve_for_D_d_Q_R (D d Q R : ℕ) 
    (h1 : D = d * Q + R) 
    (h2 : d * Q = 135) 
    (h3 : R = 2 * d) : 
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by
  sorry

end solve_for_D_d_Q_R_l411_411373


namespace right_triangle_divisible_into_2_isosceles_triangle_divisible_into_4_isosceles_triangle_divisible_into_5_isosceles_l411_411119

/-- 
  Any right triangle can be divided into 2 isosceles triangles. 
  A right triangle (RT) is defined with a right angle.
-/
theorem right_triangle_divisible_into_2_isosceles (α β : ℝ) (hα : 0 < α) (hβ : 0 < β)
  (h_sum : α + β = 90) : 
  ∃ A B C M : ℝ, 
  is_right_triangle A B C M → is_isosceles_triangle A M B ∧ is_isosceles_triangle B M C :=
sorry

/-- 
  Any triangle (T) can be divided into 4 isosceles triangles. 
  The definition follows from general triangle properties and distributions.
-/
theorem triangle_divisible_into_4_isosceles (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : α + β + γ = 180) : 
  ∃ P Q R H : ℝ, 
  is_triangle P Q R H → 
  (∃ A B C : ℝ, decompose_into_isosceles P Q R A B C = 4) :=
sorry

/-- 
  Any triangle (T) can be divided into 5 isosceles triangles. 
  This uses a series of constructions based on smaller divisions.
-/
theorem triangle_divisible_into_5_isosceles (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : α + β + γ = 180) : 
  ∃ A B C D : ℝ, 
  is_triangle A B C D → 
  (∃ E F G H I : ℝ, decompose_into_isosceles A B C D E F G H I = 5) :=
sorry

end right_triangle_divisible_into_2_isosceles_triangle_divisible_into_4_isosceles_triangle_divisible_into_5_isosceles_l411_411119


namespace range_of_angle_in_tetrahedron_l411_411068

theorem range_of_angle_in_tetrahedron (A B C D P : Point) 
  (h_tetrahedron : regular_tetrahedron A B C D)
  (h_P_BC : P ∈ segment B C) :
  ∃ θ_min θ_max, (angle (line_through A P) (line_through D C) = θ_min ∧ θ_min = π / 3) 
  ∧ (angle (line_through A P) (line_through D C) = θ_max ∧ θ_max = π / 2) 
  ∧ ∀ θ, θ ∈ set_of_angles (line_through A P) (line_through D C) → θ_min ≤ θ ∧ θ ≤ θ_max := sorry

end range_of_angle_in_tetrahedron_l411_411068


namespace symmetry_center_2tan_2x_sub_pi_div_4_l411_411492

theorem symmetry_center_2tan_2x_sub_pi_div_4 (k : ℤ) :
  ∃ (x : ℝ), 2 * (x) - π / 4 = k * π / 2 ∧ x = k * π / 4 + π / 8 :=
by
  sorry

end symmetry_center_2tan_2x_sub_pi_div_4_l411_411492


namespace part_a_part_b_l411_411848

noncomputable theory

variables
  (n : ℕ)
  (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i)
  (hprod : ∀ i, i < n → a i * a (i + n) = 1)

theorem part_a (h : 0 < n) :
  (∑ i in finset.range n, a i) / n ≥ 1 ∨ (∑ i in finset.range n, a (i + n)) / n ≥ 1 :=
sorry

theorem part_b (hn : n ≥ 2) :
  ∃ (j k : ℕ), j ≠ k ∧ j < 2 * n ∧ k < 2 * n ∧ |a j - a k| < 1 / (n - 1) :=
sorry

end part_a_part_b_l411_411848


namespace irrational_sqrt_27_l411_411034

noncomputable def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

noncomputable def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem irrational_sqrt_27 :
  let a := (22 : ℚ) / (7 : ℚ)
  let b := 0.303003
  let c := real.sqrt 27
  let d := - real.cbrt 64
  (is_irrational c ∧ is_rational a ∧ is_rational b ∧ is_rational d) :=
by
  let a := (22 : ℚ) / (7 : ℚ)
  let b := 0.303003
  let c := real.sqrt 27
  let d := - real.cbrt 64
  have ha : is_rational a := sorry
  have hb : is_rational b := sorry
  have hc : is_irrational c := sorry
  have hd : is_rational d := sorry
  exact ⟨hc, ha, hb, hd⟩

end irrational_sqrt_27_l411_411034


namespace prob_two_ones_in_twelve_dice_l411_411567

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411567


namespace adjacent_sum_correct_l411_411131

noncomputable def sum_of_adjacent_to_seven_in_circle_with_common_factors : ℕ :=
  147 -- The correct sum according to the problem's conditions and solution.

theorem adjacent_sum_correct :
  (∀ (x y : ℕ), x ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧ y ∈ {2, 4, 7, 14, 28, 49, 98, 196} →
    x ≠ y → nat.gcd x y > 1) →
  ∃ (a b : ℕ), a ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧ b ∈ {2, 4, 7, 14, 28, 49, 98, 196} ∧
    a ≠ 7 ∧ b ≠ 7 ∧ nat.gcd a 7 > 1 ∧ nat.gcd b 7 > 1 ∧ a + b = sum_of_adjacent_to_seven_in_circle_with_common_factors :=
begin
  sorry
end

end adjacent_sum_correct_l411_411131


namespace total_volume_of_quiche_l411_411861

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end total_volume_of_quiche_l411_411861


namespace mangoes_combined_l411_411685

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l411_411685


namespace yogurt_production_cost_l411_411163

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l411_411163


namespace Tom_runs_60_miles_per_week_l411_411534

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l411_411534


namespace find_p_q_sum_l411_411709

theorem find_p_q_sum (p q : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = 0 → 3 * x ^ 2 - p * x + q = 0) →
  p = 24 ∧ q = 45 ∧ p + q = 69 :=
by
  intros h
  have h3 := h 3 (by ring)
  have h5 := h 5 (by ring)
  sorry

end find_p_q_sum_l411_411709


namespace sum_of_numbers_gt_1_1_equals_3_9_l411_411984

noncomputable def sum_of_elements_gt_1_1 : Float :=
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  let numbers_gt_1_1 := List.filter (fun x => x > 1.1) numbers
  List.sum numbers_gt_1_1

theorem sum_of_numbers_gt_1_1_equals_3_9 :
  sum_of_elements_gt_1_1 = 3.9 := by
  sorry

end sum_of_numbers_gt_1_1_equals_3_9_l411_411984


namespace remainder_polynomial_division_l411_411854

theorem remainder_polynomial_division (P : Polynomial ℂ) (a b c : ℂ) (A B C : ℂ) :
  let R := fun x => A * ((x - b) * (x - c)) / ((a - b) * (a - c)) + B * ((x - a) * (x - c)) / ((b - a) * (b - c)) + C * ((x - a) * (x - b)) / ((c - a) * (c - b))
  in (P % ((Polynomial.X - a) * (Polynomial.X - b) * (Polynomial.X - c)) = R x) :=
sorry

end remainder_polynomial_division_l411_411854


namespace solution_set_of_inequality_l411_411932

theorem solution_set_of_inequality:
  {x : ℝ | (1 / x) < (1 / 2)} = {x : ℝ | x ∈ Ioi 2} ∪ {x : ℝ | x ∈ Iio 0} :=
by
  sorry

end solution_set_of_inequality_l411_411932


namespace distinct_b_i_l411_411522

variable {n : ℕ} (a : Fin n → ℤ) 
variables (b : Fin n → ℤ)

noncomputable def b_i (i : Fin n) : ℤ :=
  (a i + (Fin.range n).sum (λ k => ((k + 1) % n) * a ((i + k + 1) % n)))

theorem distinct_b_i (h_sum : (Fin.range n).sum a = 1) (ha : ∀ i : Fin n, b i = b_i a i) :
  Function.Injective b := 
  sorry

end distinct_b_i_l411_411522


namespace toenail_size_ratio_big_toenail_to_regular_toenail_ratio_l411_411784

-- Definitions related to conditions:
def big_toenail_size : ℕ  -- size of a big toenail
def regular_toenail_size : ℕ  -- size of a regular toenail
def jar_capacity : ℕ := 100 -- the jar can fit 100 regular toenails

def current_big_toenails : ℕ := 20  -- Hilary has filled the jar with 20 big toenails
def current_regular_toenails : ℕ := 40  -- Hilary has filled the jar with 40 regular toenails
def additional_regular_toenails : ℕ := 20  -- she can fit 20 more regular toenails

-- We need to prove that:
theorem toenail_size_ratio : 
  (current_big_toenails * big_toenail_size) + (current_regular_toenails * regular_toenail_size) + (additional_regular_toenails * regular_toenail_size) = jar_capacity * regular_toenail_size :=
sorry

theorem big_toenail_to_regular_toenail_ratio : 
  big_toenail_size = 2 * regular_toenail_size :=
sorry

end toenail_size_ratio_big_toenail_to_regular_toenail_ratio_l411_411784


namespace Tom_runs_60_miles_per_week_l411_411533

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l411_411533


namespace sin_thirteen_pi_over_six_l411_411277

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l411_411277


namespace slipper_cost_l411_411069

def original_price : ℝ := 50.00
def discount_rate : ℝ := 0.10
def embroidery_rate_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00

theorem slipper_cost :
  (original_price - original_price * discount_rate) + 
  (embroidery_rate_per_shoe * number_of_shoes) + 
  shipping_cost = 66.00 :=
by sorry

end slipper_cost_l411_411069


namespace function_equality_l411_411380

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l411_411380


namespace probability_two_ones_in_twelve_dice_l411_411586
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411586


namespace intersection_points_of_curves_l411_411785

theorem intersection_points_of_curves :
  (∀ θ : ℝ, ρ = 11 * cos (θ / 2) ∨ ρ = cos (θ / 2)) → 
  ∃ θ1 θ2 θ3 θ4 : ℝ, 
    (ρ θ1 = 11 * cos (θ1 / 2) ∧ ρ θ1 = cos (θ1 / 2)) ∧
    (ρ θ2 = 11 * cos (θ2 / 2) ∧ ρ θ2 = cos (θ2 / 2)) ∧
    (ρ θ3 = 11 * cos (θ3 / 2) ∧ ρ θ3 = cos (θ3 / 2)) ∧
    (ρ θ4 = 11 * cos (θ4 / 2) ∧ ρ θ4 = cos (θ4 / 2)) ∧
    θ1 ≠ θ2 ∧ θ1 ≠ θ3 ∧ θ1 ≠ θ4 ∧ θ2 ≠ θ3 ∧ θ2 ≠ θ4 ∧ θ3 ≠ θ4 :=
sorry

end intersection_points_of_curves_l411_411785


namespace egyptian_fraction_l411_411094

theorem egyptian_fraction (a b : ℤ) (h : 0 < b) (h_le : a ≤ b) : 
  ∃ (k : ℕ) (n : Fin k → ℕ), 
    (∀ j, 0 < n j) ∧ 
    (∀ i j, i ≠ j → n i ≠ n j) ∧ 
    (∑ i in Finset.range k, (1 : ℚ) / n i) = (a : ℚ) / b :=
by
  sorry

end egyptian_fraction_l411_411094


namespace average_hard_drive_doubling_l411_411795

theorem average_hard_drive_doubling (x : ℝ) 
  (h_capacity : ∀ n : ℕ, n = 50 → (0.2 * (2 ^ (n / x))) = 2050) : 
  x ≈ 4 :=
by 
  have h : 0.2 * 2 ^ (50 / x) = 2050 := h_capacity 50 rfl
  have eq1 : log (10250 : ℝ) / log 2 = 50 / x := sorry
  have eq2 : x = 50 / (log (10250 : ℝ) / log 2) := sorry
  have approx_x : x ≈ 4 := sorry
  exact approx_x

end average_hard_drive_doubling_l411_411795


namespace two_trains_cross_time_l411_411615

theorem two_trains_cross_time :
  let length_train1 := 180   -- Length of the first train in meters
  let length_train2 := 160   -- Length of the second train in meters
  let speed_train1 := 60     -- Speed of the first train in km/hr
  let speed_train2 := 40     -- Speed of the second train in km/hr
  let relative_speed := (speed_train1 + speed_train2) * (5 / 18)   -- Relative speed in m/s
  let total_distance := length_train1 + length_train2              -- Total distance to be covered in meters
  total_distance / relative_speed = 12.24 := 
by
  unfold length_train1 length_train2 speed_train1 speed_train2 relative_speed total_distance
  norm_num
  sorry  

end two_trains_cross_time_l411_411615


namespace general_formula_a_min_value_n_l411_411749

-- Definition of sequence {a_n} and sum sequence S_n such that S_n = 2a_n - n
def sequence_a (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := ∀ n : ℕ, n > 0 → S n = 2 * a n - n

-- Definition of the alternative sequence {b_n} related to {a_n}
def sequence_b (a : ℕ → ℤ) (b : ℕ → ℤ) : Prop := ∀ n : ℕ, n > 0 → b n = (2 * n + 1) * a n + (2 * n + 1)

-- Sum of the first n terms of sequence {b_n} is T_n
def sum_sequence_b (b T : ℕ → ℤ) : Prop := ∀ n : ℕ, n > 0 → 
  T n = finset.sum (finset.range n) b

-- General formula for {a_n}
theorem general_formula_a (a : ℕ → ℤ) (S : ℕ → ℤ) (h_a : sequence_a a S) : 
  ∀ n : ℕ, n > 0 → a n = 2^n - 1 := 
sorry

-- Minimum value of n satisfying the given inequality
theorem min_value_n (a b T : ℕ → ℤ) (h_a : sequence_a a (λ n, 2 * a n - n))
  (h_b : sequence_b a b) (h_T : sum_sequence_b b T) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m > 0, m < n → (T m - 2) / (2 * m - 1) < 128) ∧ (T n - 2) / (2 * n - 1) ≥ 128 :=
sorry

end general_formula_a_min_value_n_l411_411749


namespace factory_output_l411_411129

theorem factory_output (initial_output : ℝ) (percentage_increase : ℝ) :
  initial_output = 1 → percentage_increase = 0.1 →
  (∑ k in finset.range 5, (1 + percentage_increase)^(k + 1)) = 11 * ((1 + percentage_increase)^5 - 1) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end factory_output_l411_411129


namespace simplify_and_rationalize_l411_411885

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411885


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411543

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411543


namespace entry_exit_options_l411_411934

theorem entry_exit_options :
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  (total_gates * total_gates = 49) :=
by {
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  show total_gates * total_gates = 49
  sorry
}

end entry_exit_options_l411_411934


namespace find_points_l411_411120

noncomputable def f (x y z : ℝ) : ℝ := (x^2 + y^2 + z^2) / (x + y + z)

theorem find_points :
  (∃ (x₀ y₀ z₀ : ℝ), 0 < x₀^2 + y₀^2 + z₀^2 ∧ x₀^2 + y₀^2 + z₀^2 < 1 / 1999 ∧
    1.999 < f x₀ y₀ z₀ ∧ f x₀ y₀ z₀ < 2) :=
  sorry

end find_points_l411_411120


namespace Marcus_fit_pies_l411_411860

theorem Marcus_fit_pies (x : ℕ) 
(h1 : ∀ b, (7 * b - 8) = 27) : x = 5 := by
  sorry

end Marcus_fit_pies_l411_411860


namespace adjacent_integer_sum_l411_411133

theorem adjacent_integer_sum (d195 : Finset ℕ := {2, 4, 7, 14, 28, 49, 98, 196}) :
  (∃ a b ∈ d195, a ≠ b ∧ gcd a 7 > 1 ∧ gcd b 7 > 1) → a + b = 63 := 
by {
    sorry
}

end adjacent_integer_sum_l411_411133


namespace savings_of_B_l411_411631

section

variable (x y : ℕ)
variable (income_A income_B exp_A exp_B savings_A savings_B : ℕ)

-- Given conditions
def condition1 : income_B = 7200 := rfl
def condition2 : 6 * x = 7200 := by sorry
def condition3 : 3 * y = 6000 - 1800 := by sorry
def condition4 : income_A = 5 * x := by sorry
def condition5 : exp_A = 3 * y := by sorry
def condition6 : exp_B = 4 * y := by sorry
def condition7 : savings_A = 1800 := rfl

-- Define savings of B
def savings_B := income_B - exp_B

-- The theorem to prove
theorem savings_of_B :
  savings_B = 1600 :=
by sorry

end

end savings_of_B_l411_411631


namespace product_gcd_lcm_l411_411298

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l411_411298


namespace simplify_rationalize_denominator_l411_411903

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411903


namespace option_D_is_simplest_form_l411_411180

theorem option_D_is_simplest_form (b a x : ℝ) (h1 : ¬ ∃ c : ℝ, c ≠ 0 ∧ 2 = 3 * a * c) (h2 : ¬ ∃ c : ℝ, c ≠ 0 ∧ (1 - x) = (x - 1) * c) (h3 : ¬ ∃ c : ℝ, c ≠ 0 ∧ (a^2 - 1) = (a - 1) * c): 
  ¬ ∃ c : ℝ, c ≠ 0 ∧ x = (x^2 + 1) * c :=
begin
  sorry
end

end option_D_is_simplest_form_l411_411180


namespace chord_length_correct_l411_411109

noncomputable def chord_length : Prop :=
  let line_eq : ℝ → ℝ → Prop := λ x y, 3 * x - 4 * y - 9 = 0
  let circle_eq : ℝ → ℝ → Prop := λ x y, (x - 3)^2 + y^2 = 9
  ∃ (l : ℝ), (∀ (x y : ℝ), line_eq x y → circle_eq x y) ∧ (l = 6)

theorem chord_length_correct : chord_length :=
by
  sorry

end chord_length_correct_l411_411109


namespace complex_problem_l411_411762

def is_imaginary_unit (x : ℂ) : Prop := x^2 = -1

theorem complex_problem (a b : ℝ) (i : ℂ) (h1 : (a - 2 * i) / i = (b : ℂ) + i) (h2 : is_imaginary_unit i) :
  a - b = 1 := 
sorry

end complex_problem_l411_411762


namespace range_of_lambda_l411_411353

noncomputable def a : ℕ → ℝ
| 0     := 1  -- Note: Since typical sequence indices start from 1, but Lean sequences start from 0, we'll adjust accordingly
| (n+1) := a n

noncomputable def b (λ : ℝ) (n : ℕ) : ℝ :=
(n - λ) * (2^n)

def increasing_seq (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b (n+1) > b n

theorem range_of_lambda (λ : ℝ) :
    increasing_seq (b λ) ∧ b λ 0 = -λ → λ < 2 :=
sorry

end range_of_lambda_l411_411353


namespace probability_exactly_two_ones_l411_411559

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411559


namespace trapezoid_area_ratio_l411_411167

theorem trapezoid_area_ratio {P Q R S T U V W X : Type*} [EquilateralTriangle P Q R]
  (h1 : parallel S T Q R) (h2 : parallel U V Q R) (h3 : parallel W X Q R) 
  (h4 : segment_length PS = segment_length SU) 
  (h5 : segment_length SU = segment_length UW)
  (h6 : segment_length UW = segment_length WX) : 
  area_ratio (trapezoid W X Q R) (triangle P Q R) = 7 / 16 :=
sorry

end trapezoid_area_ratio_l411_411167


namespace soccer_red_cards_l411_411022

theorem soccer_red_cards (total_players yellow_carded_players : ℕ)
  (no_caution_players : total_players = 11)
  (five_no_cautions : 5 = 5)
  (players_received_yellow : yellow_carded_players = total_players - 5)
  (yellow_per_player : ∀ p, p = 6 -> 6 * 1)
  (red_card_rule : ∀ y, (yellow_carded_players * 1) = y -> y / 2 = 3) :
  ∃ red_cards : ℕ, red_cards = 3 :=
by { 
  existsi 3,
  sorry
}

end soccer_red_cards_l411_411022


namespace percentage_increase_is_20_percent_l411_411926

noncomputable def SP : ℝ := 8600
noncomputable def CP : ℝ := 7166.67
noncomputable def percentageIncrease : ℝ := ((SP - CP) / CP) * 100

theorem percentage_increase_is_20_percent : percentageIncrease = 20 :=
by
  sorry

end percentage_increase_is_20_percent_l411_411926


namespace range_of_m_l411_411742

-- Defining the conditions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Defining the main theorem
theorem range_of_m (m : ℝ) : (∀ x : ℝ, q x → p x m) ∧ ¬ (∀ x : ℝ, p x m → q x) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end range_of_m_l411_411742


namespace cyclic_quadrilateral_JKEF_l411_411443

open_locale classical

variables {A B C H E F J K : Type} [EuclideanGeometry A B C H E F J K]

-- Assuming all given conditions
variable (acute_angled_triangle : acute_angled A B C)
variable (is_orthocenter : orthocenter H A B C)
variable (foot_of_altitude_B : foot_of_altitude E B)
variable (foot_of_altitude_C : foot_of_altitude F C)
variable (midpoint_HB : midpoint J H B)
variable (midpoint_HC : midpoint K H C)

theorem cyclic_quadrilateral_JKEF :
  cyclic_quadrilateral J K E F :=
begin
  sorry
end

end cyclic_quadrilateral_JKEF_l411_411443


namespace determine_phi_l411_411769

theorem determine_phi (ω : ℝ) (h_ω : ω > 0) (ϕ : ℝ) (h_ϕ : |ϕ| < π / 2)
  (h_sym_center : ∀ k : ℤ, 
  ∃ C : ℝ, C = (k * π / 2 - π / 12) ∧
  ∀ x : ℝ, 2 * sin (ω * x + π / 6) = cos (2 * x + ϕ)) : 
  ϕ = - π / 3 :=
sorry

end determine_phi_l411_411769


namespace cosine_identity_l411_411376

theorem cosine_identity (α : ℝ) (h : sin (3 * π + α) = - 1 / 2) : cos (7 * π / 2 - α) = - 1 / 2 :=
by 
  sorry

end cosine_identity_l411_411376


namespace last_two_digits_of_sum_of_first_50_factorials_l411_411962

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l411_411962


namespace seating_arrangement_possible_iff_power_of_two_l411_411634

-- Problem Statement:
theorem seating_arrangement_possible_iff_power_of_two
  (n : ℕ) : (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (∃! s_k : ℕ, s_k = (k * (k - 1) / 2) % n)) ↔ ∃ m : ℕ, n = 2 ^ m :=
begin
  sorry
end

end seating_arrangement_possible_iff_power_of_two_l411_411634


namespace rationalize_denominator_l411_411895

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411895


namespace probability_two_ones_in_twelve_dice_l411_411587
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411587


namespace find_smallest_among_given_numbers_l411_411988

def smallest_number (l : List ℕ) : ℕ :=
  List.minimum l

theorem find_smallest_among_given_numbers :
  smallest_number [5, 8, 3, 2, 6] = 2 :=
by
  sorry

end find_smallest_among_given_numbers_l411_411988


namespace proof_f_ff_f_one_third_l411_411312

noncomputable def f : ℝ → ℝ
| x := if x > 0 then logBase 3 x
       else if x = 0 then -(2 ^ x)
       else x^2 - 1

theorem proof_f_ff_f_one_third :
  f (f (f (1 / 3))) = -1 :=
by {
  sorry
}

end proof_f_ff_f_one_third_l411_411312


namespace count_positive_integers_satisfying_properties_l411_411128

theorem count_positive_integers_satisfying_properties :
  (∃ n : ℕ, ∀ N < 2007,
    (N % 2 = 1) ∧
    (N % 3 = 2) ∧
    (N % 4 = 3) ∧
    (N % 5 = 4) ∧
    (N % 6 = 5) → n = 33) :=
by
  sorry

end count_positive_integers_satisfying_properties_l411_411128


namespace multiplier_condition_l411_411802

theorem multiplier_condition (a b : ℚ) (h : a * b ≤ b) : (b ≥ 0 ∧ a ≤ 1) ∨ (b ≤ 0 ∧ a ≥ 1) :=
by 
  sorry

end multiplier_condition_l411_411802


namespace prob_two_ones_in_twelve_dice_l411_411564

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411564


namespace sum_of_two_digit_numbers_with_odd_digits_l411_411518

-- Define a two-digit number whose both digits are odd
def isTwoDigitBothOddDigits (n : ℕ) : Prop :=
  n / 10 % 2 = 1 ∧ n % 10 % 2 = 1 ∧ 10 ≤ n ∧ n < 100

-- Define the sum of all two-digit numbers whose digits are both odd
def sumTwoDigitsOdd : ℕ :=
  ∑ n in Finset.range 100, if isTwoDigitBothOddDigits n then n else 0

-- State the theorem that the sum of all two-digit numbers whose digits are both odd is 1375
theorem sum_of_two_digit_numbers_with_odd_digits : sumTwoDigitsOdd = 1375 :=
by
  sorry

end sum_of_two_digit_numbers_with_odd_digits_l411_411518


namespace joan_seashells_initially_l411_411843

variable (mikeGave joanTotal : ℕ)

theorem joan_seashells_initially (h : mikeGave = 63) (t : joanTotal = 142) : joanTotal - mikeGave = 79 := 
by
  sorry

end joan_seashells_initially_l411_411843


namespace dice_probability_l411_411592

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411592


namespace sat_marking_problem_l411_411185

-- Define the recurrence relation for the number of ways to mark questions without consecutive markings of the same letter.
def f : ℕ → ℕ
| 0     => 1
| 1     => 2
| 2     => 3
| (n+2) => f (n+1) + f n

-- Define that each letter marking can be done in 32 different ways.
def markWays : ℕ := 32

-- Define the number of questions to be 10.
def numQuestions : ℕ := 10

-- Calculate the number of sequences of length numQuestions with no consecutive same markings.
def numWays := f numQuestions

-- Prove that the number of ways results in 2^20 * 3^10 and compute 100m + n + p where m = 20, n = 10, p = 3.
theorem sat_marking_problem :
  (numWays ^ 5 = 2 ^ 20 * 3 ^ 10) ∧ (100 * 20 + 10 + 3 = 2013) :=
by
  sorry

end sat_marking_problem_l411_411185


namespace log_eq_3_implies_x_is_200_l411_411261

theorem log_eq_3_implies_x_is_200 (x : ℝ) (h : log 10 (5 * x) = 3) : x = 200 :=
sorry

end log_eq_3_implies_x_is_200_l411_411261


namespace sum_of_digits_M_is_9_l411_411445

-- Definition for the number of divisors
def d (n : ℕ) : ℕ := (List.range (n + 1)).count (λ m, m > 0 ∧ n % m = 0)

-- Definition for the function g
def g (n : ℕ) :  ℚ := ((d n)^2 : ℚ) / (n : ℚ)^(1/4)

-- Definition for the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Hypothesis that leads to the maximum M
def M : ℕ := 432

-- The final theorem statement
theorem sum_of_digits_M_is_9 : sum_of_digits M = 9 := by
  sorry

end sum_of_digits_M_is_9_l411_411445


namespace simplify_and_rationalize_l411_411887

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411887


namespace books_on_shelf_l411_411046

theorem books_on_shelf
  (initial_action_figures : ℕ)
  (added_action_figures : ℕ)
  (extra_books : ℕ)
  (h_initial : initial_action_figures = 2)
  (h_added : added_action_figures = 4)
  (h_extra : extra_books = 4) :
  let total_action_figures := initial_action_figures + added_action_figures in
  let books := total_action_figures + extra_books in
  books = 10 := 
by
  sorry

end books_on_shelf_l411_411046


namespace find_x2_y2_l411_411718

theorem find_x2_y2 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : xy + x + y = 35) (h4 : xy * (x + y) = 360) : x^2 + y^2 = 185 := by
  sorry

end find_x2_y2_l411_411718


namespace num_divisors_exponent_1728_l411_411369

theorem num_divisors_exponent_1728 {n : ℕ} :
  let base := 1728,
      exponent := 1728 in
  n = base ^ exponent →
  let divisors := (10368 + 1) * (5184 + 1) in
  let condition := 1728 in
  (∃ d ∈ (nat.divisors n), (nat.divisors d).card = condition) →
    20 :=
begin
  sorry
end

end num_divisors_exponent_1728_l411_411369


namespace exists_prime_q_l411_411064

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q, Nat.Prime q ∧ ∀ n, ¬ (q ∣ n^p - p) := by
  sorry

end exists_prime_q_l411_411064


namespace count_divisors_with_1728_divisors_l411_411372

noncomputable def num_divisors_of_1728_exp_1728_with_1728_divisors : Nat :=
  ∑ x in (finset.range 10369).filter (λ a, (∃ b, b ∈ (finset.range 5185) ∧ (a + 1) * (b + 1) = 1728)), 1

theorem count_divisors_with_1728_divisors : 
  num_divisors_of_1728_exp_1728_with_1728_divisors = 18 := 
sorry

end count_divisors_with_1728_divisors_l411_411372


namespace total_players_in_tournament_l411_411816

-- Define the conditions of the problem
variable (n : ℕ) -- Number of players not in the weakest 15

-- Define the constraints/bounds as described in the problem
def tournament (n : ℕ) : Prop :=
  (finset.card (finset.range (n + 15)) = n + 15) ∧
  let total_points := n^2 - n + 210 in
  let actual_points := ((n + 15) * (n + 14)) / 2 in
  (total_points = actual_points)

-- Statement to prove the total number of players
theorem total_players_in_tournament : (∃ n : ℕ, tournament n ∧ (n + 15 = 36)) :=
sorry

end total_players_in_tournament_l411_411816


namespace triangle_sides_sqrt_P_l411_411441

variables {n : ℕ}
variables {a b c : ℝ}
variables {P : ℝ → ℝ}

-- Hypotheses:
-- P(x) is a polynomial of degree n (n >= 2) with non-negative coefficients
hypothesis h1 : ∃ p_n p_{n-1} ... p_0 : ℝ, ∀ x, P x = p_n * x^n + p_{n-1} * x^(n-1) + ... + p_0 ∧ ∀ i, p_i ≥ 0
-- a, b, c are the side lengths of a triangle
hypothesis h2 : a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sides_sqrt_P :
  n ≥ 2 →
  sqrt[n] (P a) + sqrt[n] (P b) > sqrt[n] (P c) ∧
  sqrt[n] (P a) + sqrt[n] (P c) > sqrt[n] (P b) ∧
  sqrt[n] (P b) + sqrt[n] (P c) > sqrt[n] (P a) :=
sorry

end triangle_sides_sqrt_P_l411_411441


namespace probability_two_ones_in_twelve_dice_l411_411546

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411546


namespace monotonic_increasing_interval_l411_411922

theorem monotonic_increasing_interval :
  ∀ x : ℝ, f(x) = 3 * x - x ^ 3 → -1 < x ∧ x < 1 → increasing_on f (-1, 1) :=
by
  sorry

end monotonic_increasing_interval_l411_411922


namespace rationalize_denominator_l411_411891

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411891


namespace hyperbola_foci_coordinates_l411_411774

theorem hyperbola_foci_coordinates 
  (a b c : ℝ)
  (h : ∀ x y : ℝ, (x / a)^2 - y^2 = 1)
  (P : (ℝ × ℝ)) 
  (P_on_hyperbola: P = (2 * real.sqrt 2, 1) ∧ (P.1 / a)^2 - (P.2)^2 = 1) 
  (a_sq_eq_4 : a ^ 2 = 4)
  (b_eq_1 : b ^ 2 = 1)
  (c_pos : c = real.sqrt (a ^ 2 + b ^ 2)):
  (real.sqrt 5, 0) ∈ {(x, y) | (x / a)^2 - y^2 = 1} ∧ (-real.sqrt 5, 0) ∈ {(x, y) | (x / a)^2 - y^2 = 1} := 
sorry

end hyperbola_foci_coordinates_l411_411774


namespace area_between_curves_l411_411107

def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

theorem area_between_curves : 
  (∫ x in 1..3, f x - g x) = 8 / 3 := 
by
  sorry

end area_between_curves_l411_411107


namespace conjugate_of_z_l411_411490

def z : ℂ := 2 + ⟨0, 1⟩

theorem conjugate_of_z : conj z = 2 - ⟨0, 1⟩ := 
by 
  sorry

end conjugate_of_z_l411_411490


namespace sequence_thirty_l411_411260

noncomputable def c : ℕ → ℤ
| 1 := 1
| 2 := 3
| (n+1) := c n * c (n-1)

theorem sequence_thirty :
  c 30 = 3 ^ 514229 :=
sorry

end sequence_thirty_l411_411260


namespace inscribed_right_triangle_exists_l411_411316

noncomputable theory

-- Definitions of the geometric entities involved.
variables (O : Point) -- center of the circle ω
variables (A B : Point) -- points inside the circle ω
variables (ω : Circle O) -- circle ω centered at O

-- Proposition: There exists a right-angled triangle inscribed in ω with its legs passing through A and B.
theorem inscribed_right_triangle_exists (A B : Point) (ω : Circle Point) :
  ∃ C : Point, on_circle C ω ∧ ∠ A C B = 90 :=
sorry

end inscribed_right_triangle_exists_l411_411316


namespace sum_positive_real_solutions_l411_411726

theorem sum_positive_real_solutions :
  let f := λ x : ℝ, 2 * sin (2 * x) * (sin (2 * x) + sin (1007 * π ^ 2 / x)) = 1 - cos (4 * x)
  in (∀ x : ℝ, 0 < x → f x → x ∈ {a : ℝ | a = (π / 2) * y ∧ y ∈ {2, 38, 106, 2014}}) →
     (∑ (x ∈ {2, 38, 106, 2014}), (π / 2) * x) = 1080 * π :=  
by
  intros f h
  have H := λ y : ℕ, ∑ (x : ℝ) in {2, 38, 106, 2014}, (π / 2) * x
  rw H
  sorry

end sum_positive_real_solutions_l411_411726


namespace range_of_reciprocal_sum_l411_411329

theorem range_of_reciprocal_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := 
sorry

end range_of_reciprocal_sum_l411_411329


namespace james_eats_three_l411_411838

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end james_eats_three_l411_411838


namespace original_number_l411_411045

variable (x : ℝ)

theorem original_number (h1 : x - x / 10 = 37.35) : x = 41.5 := by
  sorry

end original_number_l411_411045


namespace tangent_summation_l411_411197

theorem tangent_summation (α : ℝ) : 
  ∑ n in finset.range (2016 + 1).filter (λ n, 2 ≤ n) (λ n, (tan (n * α) * tan ((n - 1) * α))) = 
  (tan (2017 * α) / tan α) - 2016 :=
sorry

end tangent_summation_l411_411197


namespace bacteria_population_remain_l411_411218

theorem bacteria_population_remain 
  (initial_bacteria : ℕ)
  (time_minutes : ℕ)
  (doubling_time : ℕ)
  (removal_percentage : ℝ)
  (population : ℕ := initial_bacteria * 2^(time_minutes / doubling_time))
  (removed_population : ℕ := (removal_percentage * population).to_nat)
  (final_population : ℕ := population - removed_population) :
  initial_bacteria = 50 →
  time_minutes = 16 →
  doubling_time = 4 →
  removal_percentage = 0.25 →
  final_population = 600 :=
by
  intros
  sorry

end bacteria_population_remain_l411_411218


namespace biking_event_l411_411811

theorem biking_event :
  (let speed_carla := 12
       speed_derek := 15
       time_carla := 3.5
       time_derek := 3
       dist_carla := speed_carla * time_carla
       dist_derek := speed_derek * time_derek
   in dist_derek - dist_carla = 3) :=
by
  let speed_carla := 12
  let speed_derek := 15
  let time_carla := 3.5
  let time_derek := 3
  let dist_carla := speed_carla * time_carla
  let dist_derek := speed_derek * time_derek
  have h1 : dist_carla = 42 := by sorry
  have h2 : dist_derek = 45 := by sorry
  show dist_derek - dist_carla = 3 from by sorry

end biking_event_l411_411811


namespace sequence_sum_l411_411174

theorem sequence_sum : 
  ∑ k in Finset.range 24, if k % 2 = 0 then (3 * k + 1) else (- (3 * k + 4)) = -36 :=
by
  sorry

end sequence_sum_l411_411174


namespace simplify_and_rationalize_l411_411898

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411898


namespace sin_double_sub_l411_411311

theorem sin_double_sub (α : ℝ) (h : sin (π / 3 - α) = 1 / 3) : 
  sin (π / 6 - 2 * α) = -7 / 9 :=
sorry

end sin_double_sub_l411_411311


namespace probability_two_dice_show_1_l411_411576

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411576


namespace prime_factorization_8820_least_n_for_factor_8820_l411_411619

-- Given: the prime factorization of 8820
def factor_8820 : ℕ := 8820
theorem prime_factorization_8820 : factor_8820 = 2^2 * 3 * 5 * 7^2 := by
  sorry
-- Find least positive integer n such that 8820 divides n!
def has_factorial_divisor (n m : ℕ) : Prop := m ∣ (nat.factorial n)

theorem least_n_for_factor_8820 (n : ℕ) (h : has_factorial_divisor n factor_8820) : n = 14 := by
  sorry

end prime_factorization_8820_least_n_for_factor_8820_l411_411619


namespace total_valid_arrangements_l411_411083

open Finset

-- Define the problem conditions
def competitions : Finset String := { "volleyball", "basketball", "soccer" }
def stadiums : Finset String := { "stadium1", "stadium2", "stadium3", "stadium4" }

-- Lean statement for proving the total number of valid arrangements
theorem total_valid_arrangements : 
  (∀ (arr : competitions → stadiums), 
     (∀ s ∈ stadiums, (filter (λ c, arr c = s) competitions).card ≤ 2)) → 
  (finset.card (powerset_len 3 stadiums) - card (filter (λ arr, ∃ s ∈ stadiums, (filter (λ c, arr c = s) competitions).card = 3) (combinations 3 stadiums))) = 60 :=
sorry

end total_valid_arrangements_l411_411083


namespace distinct_values_for_D_l411_411400

variables {A B C E D : ℕ}

-- Given conditions
axiom digits_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E
axiom digit_range : ∀ x, (x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E) → 0 ≤ x ∧ x ≤ 9
axiom addition_holds : (10^4 * A + 10^3 * B + 10^2 * C + 10 * E + B) + 
                             (10^4 * B + 10^3 * C + 10^2 * E + 10 * D + A) = 
                             10^4 * D + 10^3 * B + 10^2 * D + 10 * D + D

-- The property to prove
theorem distinct_values_for_D: (set_of (λ x, (∃ A B C E, digits_distinct ∧ digit_range ∧ addition_holds ∧ x = D))).card = 7 :=
sorry

end distinct_values_for_D_l411_411400


namespace problem1_problem2_l411_411879

-- Statement for the first problem
theorem problem1 (θ : ℝ) : (sin θ - cos θ) / (tan θ - 1) = cos θ :=
sorry

-- Statement for the second problem
theorem problem2 (α : ℝ) : sin α ^ 4 - cos α ^ 4 = 2 * sin α ^ 2 - 1 :=
sorry

end problem1_problem2_l411_411879


namespace minimum_area_triangle_equation_midpoint_P_l411_411348

-- Definitions based on the conditions
def parabola (x y : ℝ) := y^2 = 4 * x

def focus : ℝ × ℝ := (1, 0)

def lines_through_focus (l1 l2 : ℝ → ℝ) (x : ℝ) := 
  ∃ k1 k2 : ℝ, k1 ≠ k2 ∧ l1 x = k1 * (x - 1) ∧ l2 x = k2 * (x - 1)

def intersection_with_parabola (l : ℝ → ℝ) := 
  ∃ P : ℝ × ℝ, ∃ x y : ℝ, l x = y ∧ parabola x y = true ∧ P = (x, y)

def midpoint (P1 P2 : ℝ × ℝ) := 
  ((P1.fst + P2.fst) / 2, (P1.snd + P2.snd) / 2)

-- Proof statments for the two parts of the problem
theorem minimum_area_triangle : ∃ P1 P2 P3 P4 : ℝ × ℝ, 
  intersection_with_parabola (λ x, x + 1) ∧ intersection_with_parabola (λ x, x - 1) → 
  let M1 := midpoint P1 P2 in 
  let M2 := midpoint P3 P4 in
  ∃ k : ℝ, ∀ (k = 1 ∨ k = -1), 
  let area := 0.5 * 4 * 2 in
  area = 4 := sorry

theorem equation_midpoint_P : ∃ P1 P2 P3 P4 : ℝ × ℝ, 
  intersection_with_parabola (λ x, x + 1) ∧ intersection_with_parabola (λ x, x - 1) → 
  let M1 := midpoint P1 P2 in 
  let M2 := midpoint P3 P4 in
  ∃ x y : ℝ, 
  let P := midpoint M1 M2 in 
  y^2 = x - 3 := sorry

end minimum_area_triangle_equation_midpoint_P_l411_411348


namespace probability_p1_eq_p2_l411_411645

theorem probability_p1_eq_p2 :
  let n := 10 in
  let p := 0.1 in
  -- Conditions
  let complete_first_collection := true in
  ∃ p_1 p_2 : ℝ, (complete_first_collection → p_1 = p_2) :=
sorry

end probability_p1_eq_p2_l411_411645


namespace max_value_of_f_on_interval_l411_411763

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem max_value_of_f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 1 :=
by
  sorry

end max_value_of_f_on_interval_l411_411763


namespace C1_ordinary_equation_C2_rectangular_min_distance_l411_411639

section problem_1

variables {α : ℝ}
def C1_parametric (α : ℝ) : (ℝ × ℝ) := (2 * sqrt 2 * cos α, 2 * sin α)

theorem C1_ordinary_equation (x y : ℝ) :
  (∃ α : ℝ, x = 2 * sqrt 2 * cos α ∧ y = 2 * sin α) ↔ (x ^ 2) / 8 + (y ^ 2) / 4 = 1 := 
sorry

end problem_1

section problem_2

variables {ρ θ : ℝ}
def C2_polar (ρ θ : ℝ) : Prop := ρ * cos θ - sqrt 2 * ρ * sin θ = 5

theorem C2_rectangular (x y : ℝ) :
  (∃ ρ θ : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ * cos θ - sqrt 2 * ρ * sin θ = 5) ↔ x - sqrt 2 * y - 5 = 0 :=
sorry

end problem_2

section problem_3

variables {α : ℝ}
def P (α : ℝ) : (ℝ × ℝ) := (2 * sqrt 2 * cos α, 2 * sin α)
def Q (x y : ℝ) : Prop := x - sqrt 2 * y - 5 = 0

noncomputable def distance_PQ (α : ℝ) : ℝ :=
(abs (2 * sqrt 2 * cos α - 2 * sqrt 2 * sin α - 5)) / sqrt 3

theorem min_distance : 
  ∃ α : ℝ, cos (α + π / 4) = 1 ∧ distance_PQ α = sqrt 3 / 3 :=
sorry

end problem_3

end C1_ordinary_equation_C2_rectangular_min_distance_l411_411639


namespace distance_between_parallel_lines_l411_411918

theorem distance_between_parallel_lines :
  let L1 := λ (x y : ℝ), 3 * x - 4 * y + 2 = 0
  let L2 := λ (x y : ℝ), 6 * x - 8 * y + 14 = 0
  let d := 1
  ∀ (x y : ℝ), L1 x y → L2 x y → (d = (|7 - 2|) / (real.sqrt (3^2 + (-4)^2))) :=
begin
  sorry
end

end distance_between_parallel_lines_l411_411918


namespace inequality_solution_l411_411267

theorem inequality_solution (x : ℝ) : 3 * x^2 - 8 * x + 3 < 0 ↔ (1 / 3 < x ∧ x < 3) := by
  sorry

end inequality_solution_l411_411267


namespace remove_percentage_of_pollutants_l411_411207

noncomputable def pollutant_filtration_time (P0 : ℝ) (t k : ℝ) : ℝ := P0 * (real.exp (-k * t))

theorem remove_percentage_of_pollutants (P0 : ℝ) :
  ∀ t : ℝ, 
  let P := P0 * real.exp (-t * (- (real.log 0.9) / 5)) in
  P0 * real.exp (-5 * (- (real.log 0.9) / 5)) = 0.9 * P0 →
  P0 * real.exp (-t * (- (real.log 0.9) / 5)) = 0.271 * P0 →
  t = 15 :=
by 
intros t P condition_1 condition_2
sorry

end remove_percentage_of_pollutants_l411_411207


namespace proof_integral_solution_l411_411705

noncomputable def integral_solution (x y z : ℤ) : Prop :=
  z^x = y^(2*x) ∧ 2^z = 4 * 8^x ∧ x + y + z = 18

theorem proof_integral_solution : 
  integral_solution 5 4 17 := by
  sorry

end proof_integral_solution_l411_411705


namespace prob_two_ones_in_twelve_dice_l411_411565

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l411_411565


namespace f_neg_2_l411_411740

-- Define f(x) as given
def f (x : ℝ) : ℝ := if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- State the theorem we need to prove
theorem f_neg_2 : f (-2) = -1 :=
by
  sorry

end f_neg_2_l411_411740


namespace area_of_quadrilateral_l411_411456

/-- The area of the quadrilateral defined by the system of inequalities is 15/7. -/
theorem area_of_quadrilateral : 
  (∃ (x y : ℝ), 3 * x + 2 * y ≤ 6 ∧ x + 3 * y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0) →
  (∃ (area : ℝ), area = 15 / 7) :=
by
  sorry

end area_of_quadrilateral_l411_411456


namespace max_planes_from_15_points_l411_411977

-- Define the problem conditions
def no_four_points_coplanar (P : set (EuclideanSpace ℝ (fin 3))) : Prop :=
  ∀ (a b c d : EuclideanSpace ℝ (fin 3)), {a, b, c, d} ⊆ P → ¬ coplanar {a, b, c, d}

-- Statement of the problem
theorem max_planes_from_15_points (P : set (EuclideanSpace ℝ (fin 3))) (hP : P.card = 15) (h_nfc : no_four_points_coplanar P) : ∃ (n : ℕ), n = 455 := by
  sorry

end max_planes_from_15_points_l411_411977


namespace range_of_a_l411_411806

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - a > 0 ∧ 3 * x - 4 < 5) -> False) ↔ (a ≥ 6) :=
by
  sorry

end range_of_a_l411_411806


namespace probability_exactly_two_ones_l411_411556

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411556


namespace dice_probability_l411_411591

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411591


namespace range_of_a_l411_411770

theorem range_of_a (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = (1 / 2) * x + (1 / 2) * (a * sin x))
  (hg : ∀ x, g x = 2 * (f x) - h x)
  (hh : ∀ x, h x = (1 / 3) * sin (2 * x))
  (hg_monotone : ∀ x, g' x > 0)
  : - (1 / 3) ≤ a ∧ a ≤ (1 / 3)
  := sorry

end range_of_a_l411_411770


namespace max_value_of_m_l411_411768

noncomputable def f (x : ℝ) (b : ℝ) :=
  x ^ 2 + b * x + 1

theorem max_value_of_m (b : ℝ) (t : ℝ) (l m : ℝ) :
  (∀ x, f x b = f (-x) b ↔ f (x + 1) b) →
  (∃ t, ∀ x, l ≤ x ∧ x ≤ m → f (x + t) b ≤ x) →
  m ≤ 3 :=
by sorry

end max_value_of_m_l411_411768


namespace yogurt_cost_l411_411158

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l411_411158


namespace pyramid_addition_totals_l411_411227

theorem pyramid_addition_totals 
  (initial_faces : ℕ) (initial_edges : ℕ) (initial_vertices : ℕ)
  (first_pyramid_new_faces : ℕ) (first_pyramid_new_edges : ℕ) (first_pyramid_new_vertices : ℕ)
  (second_pyramid_new_faces : ℕ) (second_pyramid_new_edges : ℕ) (second_pyramid_new_vertices : ℕ)
  (cancelling_faces_first : ℕ) (cancelling_faces_second : ℕ) :
  initial_faces = 5 → 
  initial_edges = 9 → 
  initial_vertices = 6 → 
  first_pyramid_new_faces = 3 →
  first_pyramid_new_edges = 3 →
  first_pyramid_new_vertices = 1 →
  second_pyramid_new_faces = 4 →
  second_pyramid_new_edges = 4 →
  second_pyramid_new_vertices = 1 →
  cancelling_faces_first = 1 →
  cancelling_faces_second = 1 →
  initial_faces + first_pyramid_new_faces - cancelling_faces_first 
  + second_pyramid_new_faces - cancelling_faces_second 
  + initial_edges + first_pyramid_new_edges + second_pyramid_new_edges
  + initial_vertices + first_pyramid_new_vertices + second_pyramid_new_vertices 
  = 34 := by sorry

end pyramid_addition_totals_l411_411227


namespace ratio_of_perimeters_l411_411483

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l411_411483


namespace solution_l411_411382

noncomputable def find_x (x : ℝ) : Prop :=
  0.16 * 0.40 * x = 6

theorem solution : ∃ x : ℝ, find_x x ∧ x = 93.75 := by
  have h : 0.16 * 0.40 = 0.064 := by norm_num
  use 93.75
  -- Proof to show that 0.064 * 93.75 = 6
  calc
    0.064 * 93.75 = 6 : by norm_num
  sorry

end solution_l411_411382


namespace average_rainfall_l411_411223

-- We define the given conditions as separate definitions.
def first_30_min_rain : ℝ := 5
def next_30_min_rain : ℝ := first_30_min_rain / 2
def next_hour_rain : ℝ := 0.5

-- The total rainfall over the duration of the storm.
def total_rainfall : ℝ := first_30_min_rain + next_30_min_rain + next_hour_rain

-- The total duration of the storm in hours.
def total_duration : ℝ := 2 -- 30 minutes + 30 minutes + 1 hour

-- We now write the theorem statement to prove that the average rainfall total is 4 inches per hour.
theorem average_rainfall : (total_rainfall / total_duration) = 4 := by
  -- All the steps for the proof would go here.
  sorry

end average_rainfall_l411_411223


namespace mirror_side_length_l411_411670

theorem mirror_side_length (width length : ℝ) (area_wall : ℝ) (area_mirror : ℝ) (side_length : ℝ) 
  (h1 : width = 28) 
  (h2 : length = 31.5) 
  (h3 : area_wall = width * length)
  (h4 : area_mirror = area_wall / 2) 
  (h5 : area_mirror = side_length ^ 2) : 
  side_length = 21 := 
by 
  sorry

end mirror_side_length_l411_411670


namespace distance_PF_l411_411028

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus : ℝ × ℝ := (1, 0)

def directrix (x : ℝ) : Prop := x = -1

def point_on_parabola (x y : ℝ) : Prop := parabola x y

def perpendicular_to_directrix (y : ℝ) : ℝ × ℝ := (-1, y)

def slope_of_AF (y : ℝ) : ℝ := (y - focus.snd) / (focus.fst - (-1))

-- Prop statement in Lean 4
theorem distance_PF (x y : ℝ)
  (h1 : point_on_parabola x y)
  (h2 : slope_of_AF y = - Real.sqrt 3) :
  Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 4 := by
  sorry

end distance_PF_l411_411028


namespace range_a_l411_411852

def p (a : ℝ) : Prop :=
∀ x, (x ∈ (-∞ : ℝ, -2) ∪ (2, ∞ : ℝ) → 
  (x^2 - 4) * (x - a) is strictly increasing)

def q (a : ℝ) : Prop :=
∀ x, (∫ t in 0..x, (2 * t - 2) > a)

theorem range_a (a : ℝ) : 
  (p a ∧ ¬ q a ∧ (-1 ≤ a ∧ a ≤ 2)) ∨ 
  (¬ p a ∧ q a ∧ a < -2) ↔
  a ∈ (-∞, -2) ∪ [-1, 2] :=
sorry

end range_a_l411_411852


namespace range_of_m_for_two_distinct_roots_l411_411764

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (π/3 * x + π/6) + 2

theorem range_of_m_for_two_distinct_roots (a m : ℝ) (h : 1 ≤ a ∧ a < 2) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 < m ∧ 0 ≤ x2 ∧ x2 < m ∧ x1 ≠ x2 ∧ f x1 - a = 2 ∧ f x2 - a = 2)
  ↔ (2 < m ∧ m ≤ 6) :=
sorry

end range_of_m_for_two_distinct_roots_l411_411764


namespace probability_two_ones_in_twelve_dice_l411_411551

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411551


namespace dice_probability_l411_411597

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411597


namespace determine_S_l411_411910

theorem determine_S :
  (∃ k : ℝ, (∀ S R T : ℝ, R = k * (S / T)) ∧ (∃ S R T : ℝ, R = 2 ∧ S = 6 ∧ T = 3 ∧ 2 = k * (6 / 3))) →
  (∀ S R T : ℝ, R = 8 ∧ T = 2 → S = 16) :=
by
  sorry

end determine_S_l411_411910


namespace product_lcm_gcd_eq_108_l411_411291

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem product_lcm_gcd_eq_108 (a b : ℕ) (h1 : a = 12) (h2 : b = 9) :
  (lcm a b) * (Nat.gcd a b) = 108 := by
  rw [h1, h2] -- replace a and b with 12 and 9
  have lcm_12_9 : lcm 12 9 = 36 := sorry -- find the LCM of 12 and 9
  have gcd_12_9 : Nat.gcd 12 9 = 3 := sorry -- find the GCD of 12 and 9
  rw [lcm_12_9, gcd_12_9]
  norm_num -- simplifies the multiplication
  exact eq.refl 108

end product_lcm_gcd_eq_108_l411_411291


namespace consecutive_primes_sum_is_product_of_three_primes_l411_411063

open Nat

theorem consecutive_primes_sum_is_product_of_three_primes (p q : ℕ) (hp : Prime p) (hq : Prime q) (hq2 : 2 < q) 
    (consec : ∀ r, Prime r → r > p → r = q) : ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ p + q = a * b * c :=
by
  sorry

end consecutive_primes_sum_is_product_of_three_primes_l411_411063


namespace t_range_l411_411830

def seq_a (n : ℕ) : ℝ :=
if n = 1 then 1/2 else
let a_n := seq_a (n - 1) in (n - 1) * a_n / (n * ((n - 1) * a_n + 1))

theorem t_range (t : ℝ) : (∀ n : ℕ, 0 < n → (3 / (n^2 : ℝ) + 1 / n + t * seq_a n) ≥ 0) → t ≥ -15 / 2 :=
sorry

end t_range_l411_411830


namespace find_c_l411_411383

theorem find_c (c : ℝ) :
  (∃ (infinitely_many_y : ℝ → Prop), (∀ y, infinitely_many_y y ↔ 3 * (5 + 2 * c * y) = 18 * y + 15))
  → c = 3 :=
by
  sorry

end find_c_l411_411383


namespace cost_of_blue_pill_l411_411253

/-
Statement:
Bob takes two blue pills and one orange pill each day for three weeks.
The cost of a blue pill is $2 more than an orange pill.
The total cost for all pills over the three weeks amounts to $966.
Prove that the cost of one blue pill is $16.
-/

theorem cost_of_blue_pill (days : ℕ) (total_cost : ℝ) (cost_orange : ℝ) (cost_blue : ℝ) 
  (h1 : days = 21) 
  (h2 : total_cost = 966) 
  (h3 : cost_blue = cost_orange + 2) 
  (daily_pill_cost : ℝ)
  (h4 : daily_pill_cost = total_cost / days)
  (h5 : daily_pill_cost = 2 * cost_blue + cost_orange) :
  cost_blue = 16 :=
by
  sorry

end cost_of_blue_pill_l411_411253


namespace john_class_size_l411_411812

theorem john_class_size (b w : ℕ) (h1 : b = 29) (h2 : w = 29) : b + w + 1 = 59 :=
by
  -- Define the assumptions based on the given conditions
  have john_is_30_best : 30 = b + 1, from by rw [h1]
  have john_is_30_worst : 30 = w + 1, from by rw [h2]
  -- Prove the class size
  rw [h1, h2]
  exact rfl

end john_class_size_l411_411812


namespace rongrong_bike_speed_l411_411946

theorem rongrong_bike_speed :
  ∃ (x : ℝ), (15 / x - 15 / (4 * x) = 45 / 60) → x = 15 :=
by
  sorry

end rongrong_bike_speed_l411_411946


namespace ratio_of_perimeters_l411_411484

theorem ratio_of_perimeters (A1 A2 : ℝ) (h : A1 / A2 = 16 / 81) : 
  let s1 := real.sqrt A1 
  let s2 := real.sqrt A2 
  (4 * s1) / (4 * s2) = 4 / 9 :=
by {
  sorry
}

end ratio_of_perimeters_l411_411484


namespace sum_of_reciprocals_eq_two_l411_411523

theorem sum_of_reciprocals_eq_two (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_eq_two_l411_411523


namespace rationalize_denominator_sqrt_l411_411087

theorem rationalize_denominator_sqrt (sqrt_eq : Real.Sqrt (8) = 2 * Real.Sqrt (2)) :
  Real.Sqrt (3 / 8) = Real.Sqrt (6) / 4 :=
by
  sorry

end rationalize_denominator_sqrt_l411_411087


namespace range_of_k_for_quadratic_inequality_l411_411804

theorem range_of_k_for_quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
  sorry

end range_of_k_for_quadratic_inequality_l411_411804


namespace acres_left_untouched_l411_411205

def total_acres := 65057
def covered_acres := 64535

theorem acres_left_untouched : total_acres - covered_acres = 522 :=
by
  sorry

end acres_left_untouched_l411_411205


namespace proof_problem_l411_411005

-- Define the angles α and β as acute angles and the given condition
variables {α β : ℝ}
variables (hα1 : 0 < α) (hα2 : α < π / 2)
variables (hβ1 : 0 < β) (hβ2 : β < π / 2)
variables (h : (Real.sin α + Real.cos α) * (Real.sin β + Real.cos β) = 2)

-- Define the target expression
def expr : ℝ := (Real.sin (2 * α) + Real.cos (3 * β))^2 + (Real.sin (2 * β) + Real.cos (3 * α))^2

-- State the theorem
theorem proof_problem : expr = 3 - 2 * Real.sqrt 2 :=
sorry  -- Proof goes here

end proof_problem_l411_411005


namespace find_m_plus_c_l411_411125

theorem find_m_plus_c : 
  ∃ m c : ℝ, 
    ((∀ x y x' y' : ℝ, (x', y') = ((1 - 2 * (m / (1 + m^2))) * x + (2 * (m^2) / (1 + m^2)) * y + (2 * c * m / (1 + m^2))), (2 * c / (1 + m^2)) * y) ∧ 
      (x = -2) ∧ (y = 0) ∧ (x' = 6) ∧ (y' = 4)) → 
    m + c = 4 :=
begin
  sorry
end

end find_m_plus_c_l411_411125


namespace area_increase_of_concentric_circles_l411_411527

theorem area_increase_of_concentric_circles :
  let r_outer := 8
  let r_inner := 5
  let r_outer_new := 8 * 1.80
  let r_inner_new := 5 * 0.60
  let area_annulus (r1 r2 : ℝ) := π * r1^2 - π * r2^2
  let percent_increase (A_old A_new : ℝ) := ((A_new - A_old) / A_old) * 100
  percent_increase (area_annulus r_outer r_inner) (area_annulus r_outer_new r_inner_new) ≈ 408.7 := 
by
  sorry

end area_increase_of_concentric_circles_l411_411527


namespace min_ratio_of_areas_l411_411171

theorem min_ratio_of_areas
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < 2) (h4 : b < 2) :
  let ratio := (a + b) / (2 * a * b) in 
  (∃ a b : ℝ, a = b → ratio = Real.sqrt 2) :=
by repeat { sorry }

end min_ratio_of_areas_l411_411171


namespace distance_AB_l411_411503

/-- The distance between two points (4, a) and (5, b) is sqrt(2), given that the line passing through these points
is parallel to the line y = x + m. -/
theorem distance_AB (a b m : ℝ) (h_slope: b - a = 1) : 
  real.sqrt ((5 - 4)^2 + (b - a)^2) = real.sqrt 2 :=
by
  -- substituting the condition h_slope
  have h : (b - a) = 1 := h_slope,
  sorry

end distance_AB_l411_411503


namespace solve_ode_l411_411478

noncomputable def x (t : ℝ) : ℝ :=
  -((1 : ℝ) / 18) * Real.exp (-t) +
  (25 / 54) * Real.exp (5 * t) -
  (11 / 27) * Real.exp (-4 * t)

theorem solve_ode :
  ∀ t : ℝ, 
    (deriv^[2] x t) - (deriv x t) - 20 * x t = Real.exp (-t) ∧
    x 0 = 0 ∧
    (deriv x 0) = 4 :=
by
  sorry

end solve_ode_l411_411478


namespace S_is_empty_l411_411442

open Complex

def S := {z : ℂ | (2 + 5 * I) * z ∈ ℝ ∧ z.re = 2 * z.im}

theorem S_is_empty : S = ∅ :=
by
  -- Proof goes here
  sorry

end S_is_empty_l411_411442


namespace simplify_rationalize_denominator_l411_411907

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l411_411907


namespace exist_right_triangle_l411_411318

open Set

structure Circle (center : Point) (radius : ℝ) :=
(mem : ∀ P : Point, dist center P = radius)

variables {P : Type} [MetricSpace P] [EuclideanSpace P]

def isRightTriangleInscribed (C : Circle) (A B : Point) : Prop :=
  ∃ C : Point, C ∈ C ∧
               angle A C B = 90

theorem exist_right_triangle (C : Circle) (A B : Point) (hA : A ∈ C) (hB : B ∈ C) :
  isRightTriangleInscribed C A B :=
sorry

end exist_right_triangle_l411_411318


namespace distance_from_Q_to_EH_l411_411480

/-- Square EFGH has side lengths of 6, and N is the midpoint of GH. 
A circle with radius 3 and center N intersects a circle with radius 6 and center E at points Q and H.
What is the distance from Q to line EH? -/
theorem distance_from_Q_to_EH :
  let E := (0 : ℝ, 6 : ℝ),
      F := (6 : ℝ, 6 : ℝ),
      G := (6 : ℝ, 0 : ℝ),
      H := (0 : ℝ, 0 : ℝ),
      N := (3 : ℝ, 0 : ℝ),
      Q := (24/5 : ℝ, 12/5 : ℝ) in
  dist Q (0, 6) = 18/5 := sorry

end distance_from_Q_to_EH_l411_411480


namespace trig_identity_l411_411224

theorem trig_identity (α : ℝ) : 
  (sin α)^2 + (cos (30 * (real.pi / 180) - α))^2 - (sin α) * cos (30 * (real.pi / 180) - α) = 3 / 4 :=
sorry

end trig_identity_l411_411224


namespace avg_weight_correct_l411_411525

-- Define the conditions
def num_students_A : ℕ := 30
def num_students_B : ℕ := 20
def avg_weight_A : ℝ := 40
def avg_weight_B : ℝ := 35

-- Define the total weight calculations
def total_weight_A : ℝ := avg_weight_A * num_students_A
def total_weight_B : ℝ := avg_weight_B * num_students_B
def total_weight_class : ℝ := total_weight_A + total_weight_B

-- Define the total number of students in the whole class
def total_students_class : ℕ := num_students_A + num_students_B

-- Define the average weight calculation for the whole class
def avg_weight_class : ℝ := total_weight_class / total_students_class

-- Proposition: The average weight of the whole class is 38 kg
theorem avg_weight_correct : avg_weight_class = 38 := by
  sorry

end avg_weight_correct_l411_411525


namespace probability_of_two_ones_in_twelve_dice_rolls_l411_411544

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l411_411544


namespace problem_solution_l411_411032

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4 * x^2 + 4 * x) / (x^3 + x^2 - 2 * x)

theorem problem_solution :
  let a := 1 in
  let b := 2 in
  let c := 1 in
  let d := 0 in
  a + 2 * b + 3 * c + 4 * d = 8 :=
by {
  let a := 1,
  let b := 2,
  let c := 1,
  let d := 0,
  show a + 2 * b + 3 * c + 4 * d = 8,
  calc
    a + 2 * b + 3 * c + 4 * d
    = 1 + 2 * 2 + 3 * 1 + 4 * 0 : by refl
  ... = 8 : by refl
}

end problem_solution_l411_411032


namespace tan_theta_value_l411_411824

theorem tan_theta_value (α θ : ℝ) (h1: α + θ = (3 * π) / 4) (h2: tan α = 4) 
(h3: tan (α + θ) = -1) : tan θ = 5 / 3 :=
sorry

end tan_theta_value_l411_411824


namespace fraction_zero_value_l411_411004

theorem fraction_zero_value (x : ℝ) (h : (3 - x) ≠ 0) : (x+2)/(3-x) = 0 ↔ x = -2 := by
  sorry

end fraction_zero_value_l411_411004


namespace nearest_multiple_to_457_divisible_by_11_l411_411191

theorem nearest_multiple_to_457_divisible_by_11 : ∃ n : ℤ, (n % 11 = 0) ∧ (abs (457 - n) = 5) :=
by
  sorry

end nearest_multiple_to_457_divisible_by_11_l411_411191


namespace curve_bounded_by_disk_l411_411699

theorem curve_bounded_by_disk (C : set (ℝ × ℝ)) 
  (hC_closed : is_closed C)
  (hC_dist : ∀ P Q ∈ C, dist P Q < 1) : 
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, radius = 1 / real.sqrt 3 ∧ ∀ P ∈ C, dist P center <= radius :=
sorry

end curve_bounded_by_disk_l411_411699


namespace confidence_difference_in_hygiene_habits_risk_ratio_formula_risk_ratio_calculated_l411_411656

noncomputable section

def K_squared (n a b c d : ℕ) : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

def P_A_given_B := 40 / 100.0
def P_A_given_not_B := 10 / 100.0
def P_not_A_given_B := 1 - P_A_given_B
def P_not_A_given_not_B := 1 - P_A_given_not_B

def R : ℝ := (P_A_given_B / P_not_A_given_B) * (P_not_A_given_not_B / P_A_given_not_B)

theorem confidence_difference_in_hygiene_habits :
  let k := K_squared 200 40 10 60 90
  in k > 6.635 := 
by
  sorry

theorem risk_ratio_formula : 
  R = (P_A_given_B / P_not_A_given_B) * (P_not_A_given_not_B / P_A_given_not_B) := 
by
  sorry

theorem risk_ratio_calculated :
  R = 6 :=
by
  sorry

end confidence_difference_in_hygiene_habits_risk_ratio_formula_risk_ratio_calculated_l411_411656


namespace goldfish_graph_finite_set_of_points_l411_411360

-- Define the cost function for goldfish including the setup fee
def cost (n : ℕ) : ℝ := 20 * n + 5

-- Define the condition
def n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

-- The Lean statement to prove the nature of the graph
theorem goldfish_graph_finite_set_of_points :
  ∀ n ∈ n_values, ∃ k : ℝ, (k = cost n) :=
by
  sorry

end goldfish_graph_finite_set_of_points_l411_411360


namespace convert_142_to_base7_l411_411706

-- Definition of the function to convert to base 7
def convertToBase7 (n : ℕ) : ℕ :=
  let digits := List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

theorem convert_142_to_base7 : convertToBase7 142 = 262 :=
by
  -- Proof steps would go here, but for now, we have 'sorry'
  sorry

end convert_142_to_base7_l411_411706


namespace james_writes_pages_per_hour_l411_411043

theorem james_writes_pages_per_hour (hours_per_night : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) (total_hours : ℕ) :
  hours_per_night = 3 → 
  days_per_week = 7 → 
  weeks = 7 → 
  total_pages = 735 → 
  total_hours = 147 → 
  total_hours = hours_per_night * days_per_week * weeks → 
  total_pages / total_hours = 5 :=
by sorry

end james_writes_pages_per_hour_l411_411043


namespace ratio_area_triangle_to_circumcircle_l411_411835

-- Define the conditions as hypotheses
variable {A B C H K E : Point}
variable (ABC : Triangle A B C)
variable (BE : Line B E)
variable (AH : Line A H)
variable (angle_BAC : Angle B A C = ∠B A C)
variable (ACB_30_deg : ∠A C B = 30 °)
variable (BK_KE_ratio : Ratio (Segment B K) (Segment K E) = 2)

-- Define the areas of the shapes
noncomputable def area_triangle_BCE : ℝ := sorry
noncomputable def area_circumcircle_BCE : ℝ := sorry

-- Statement of the theorem
theorem ratio_area_triangle_to_circumcircle : 
  (area_triangle_BCE / area_circumcircle_BCE) = sqrt 3 / (14 * real.pi) :=
sorry

end ratio_area_triangle_to_circumcircle_l411_411835


namespace total_days_2010_to_2014_l411_411786

theorem total_days_2010_to_2014 : 
  let d2010 := 365 in
  let d2011 := 365 in
  let d2012 := 366 in
  let d2013 := 365 in
  let d2014 := 365 in
  d2010 + d2011 + d2012 + d2013 + d2014 = 1826
:= by
  sorry

end total_days_2010_to_2014_l411_411786


namespace second_plaque_weight_l411_411674

theorem second_plaque_weight
  (side1 : ℝ) (weight1 : ℝ) (side2 : ℝ) (weight2 : ℝ) (height: ℝ) (pi: ℝ) : 
  side1 = 6 → 
  weight1 = 24 → 
  side2 = 8 → 
  height = (side2 * real.sqrt 3) / 2 → 
  let area1 := (side1^2 * real.sqrt 3) / 4, 
      area2 := (side2^2 * real.sqrt 3) / 4, 
      area_circle := pi * ((height / 2)^2),
      area_plaque := area2 - area_circle in 
  weight2 = 27 :=
begin
  intros h_side1 h_weight1 h_side2 h_height,
  let area1 := (side1^2 * real.sqrt 3) / 4,
  let area2 := (side2^2 * real.sqrt 3) / 4,
  let area_circle := pi * ((height / 2)^2),
  let area_plaque := area2 - area_circle,
  let calculated_weight := (area_plaque / area1) * weight1,
  have approx_weight2 : calculated_weight ≈ 27,
  sorry, -- Skip the proof
end

end second_plaque_weight_l411_411674


namespace product_of_two_numbers_ratio_l411_411953

theorem product_of_two_numbers_ratio {x y : ℝ}
  (h1 : x + y = (5/3) * (x - y))
  (h2 : x * y = 5 * (x - y)) :
  x * y = 56.25 := sorry

end product_of_two_numbers_ratio_l411_411953


namespace negation_equiv_l411_411753

open Classical

-- Proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 = 0

-- Negation of proposition p
def neg_p : Prop := ∀ x : ℝ, x^2 - x + 1 ≠ 0

-- Statement to prove the equivalence of the negation of p and neg_p
theorem negation_equiv :
  ¬p ↔ neg_p := 
sorry

end negation_equiv_l411_411753


namespace inches_repaired_before_today_l411_411203

-- Definitions and assumptions based on the conditions.
def total_inches_repaired : ℕ := 4938
def inches_repaired_today : ℕ := 805

-- Target statement that needs to be proven.
theorem inches_repaired_before_today : total_inches_repaired - inches_repaired_today = 4133 :=
by
  sorry

end inches_repaired_before_today_l411_411203


namespace simplify_tax_suitable_for_leonid_l411_411428

structure BusinessSetup where
  sellsFlowers : Bool
  noPriorExperience : Bool
  worksIndependently : Bool

def LeonidSetup : BusinessSetup := {
  sellsFlowers := true,
  noPriorExperience := true,
  worksIndependently := true
}

def isSimplifiedTaxSystemSuitable (setup : BusinessSetup) : Prop :=
  setup.sellsFlowers = true ∧ setup.noPriorExperience = true ∧ setup.worksIndependently = true

theorem simplify_tax_suitable_for_leonid (setup : BusinessSetup) :
  isSimplifiedTaxSystemSuitable setup := by
  sorry

#eval simplify_tax_suitable_for_leonid LeonidSetup

end simplify_tax_suitable_for_leonid_l411_411428


namespace number_of_cars_l411_411078

/--
Given the distance between the first car and the last car is 242 meters,
and the distance between consecutive cars is 5.5 meters,
prove that the number of cars on the street is 45.
-/
theorem number_of_cars (d_total : ℝ) (d_between : ℝ) (h1 : d_total = 242) (h2 : d_between = 5.5) : 
  (d_total / d_between).toNat + 1 = 45 :=
by 
  sorry

end number_of_cars_l411_411078


namespace adrian_gets_3_contacts_l411_411677

variable (cost1 : ℝ) (contacts1 : ℝ) (cost2 : ℝ) (contacts2 : ℝ) 

-- Defining conditions for the boxes
def box1 : Prop := cost1 = 25 ∧ contacts1 = 50
def box2 : Prop := cost2 = 33 ∧ contacts2 = 99

-- The cost per contact for each box
noncomputable def cost_per_contact1 : ℝ := contacts1 / cost1
noncomputable def cost_per_contact2 : ℝ := contacts2 / cost2

-- Adrian chooses the box with the smaller cost per contact
def adrian_choice : Prop := (cost_per_contact1 ≤ cost_per_contact2 → cost_per_contact2) ∧ (cost_per_contact2 < cost_per_contact1 → cost_per_contact2)

-- Prove that Adrian will get 3 contacts per dollar in the chosen box
theorem adrian_gets_3_contacts (h1 : box1) (h2 : box2) : 
  adrian_choice cost1 contacts1 cost2 contacts2 → cost_per_contact2 = 3 := 
by
  sorry

end adrian_gets_3_contacts_l411_411677


namespace part1_part2_l411_411339

noncomputable def f (a x : ℝ) := (a * x^2 - x + a + 1) * real.exp x
noncomputable def f_deriv (a x : ℝ) := deriv (λ x : ℝ, (a * x^2 - x + a + 1) * real.exp x) x

theorem part1 (a : ℝ) (h : f_deriv a 1 = 0) : a = 1 / 4 :=
by sorry

theorem part2 (a : ℝ) (h : a ∈ set.Ioi 0) (h2 : ∀ x : ℝ, f_deriv a (x1) = 0 → x1 ∈ set.Ioi 0 → f_deriv a (x2) = 0 → x2 ∈ set.Ioi 0 → x1 ≠ x2) : a < 1 / 4 :=
by sorry

end part1_part2_l411_411339


namespace probability_two_ones_in_twelve_dice_l411_411550

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411550


namespace simplify_expression_l411_411884

theorem simplify_expression (w : ℝ) : 3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 :=
by
  sorry

end simplify_expression_l411_411884


namespace probability_two_ones_in_twelve_dice_l411_411549

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l411_411549


namespace gcd_lcm_product_l411_411295

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_product (a b : ℕ) : gcd a b * lcm a b = a * b :=
begin
  apply Nat.gcd_mul_lcm,
end

example : gcd 12 9 * lcm 12 9 = 108 :=
by
  have h : gcd 12 9 * lcm 12 9 = 12 * 9 := gcd_lcm_product 12 9
  rw [show 12 * 9 = 108, by norm_num] at h
  exact h

end gcd_lcm_product_l411_411295


namespace price_Y_half_filled_price_Y_filled_to_Z_compare_prices_of_Y_l411_411872

noncomputable def V (r h : ℝ) := π * r^2 * h

constant r_X h_X : ℝ
constant price_X : ℝ := 2
constant price_Z : ℝ := 5

axiom positive_radius : r_X > 0
axiom positive_height : h_X > 0

def price_per_unit_volume_X : ℝ := price_X / V r_X h_X

def radius_Y := 4 * r_X
def height_Y := 4 * h_X

def radius_Z := 2 * r_X
def height_Z := 2 * h_X

theorem price_Y_half_filled : price_per_unit_volume_X * (V radius_Y height_Y / 2) = 64 := sorry

theorem price_Y_filled_to_Z : price_per_unit_volume_X * V radius_Y height_Z = 64 := sorry

theorem compare_prices_of_Y : price_per_unit_volume_X * (V radius_Y height_Y / 2) = price_per_unit_volume_X * V radius_Y height_Z := sorry

end price_Y_half_filled_price_Y_filled_to_Z_compare_prices_of_Y_l411_411872


namespace point_on_parabola_l411_411124

theorem point_on_parabola (c m n x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : x1^2 + 2*x1 + c = 0)
  (hx2 : x2^2 + 2*x2 + c = 0)
  (hp : n = m^2 + 2*m + c)
  (hn : n < 0) :
  x1 < m ∧ m < x2 :=
sorry

end point_on_parabola_l411_411124


namespace matrix_N_solution_l411_411289

def A : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![2, -5],
    ![4, -3]]

def B : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![ -20, -8],
    ![ 12, 4]]

def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![ 46/7, -58/7],
    ![-26/7, 34/7]]

theorem matrix_N_solution : 
  N ⬝ A = B := by sorry

end matrix_N_solution_l411_411289


namespace probability_two_ones_in_twelve_dice_l411_411590
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411590


namespace interval_of_monotonic_increase_l411_411342

def f (ω x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

theorem interval_of_monotonic_increase (ω : ℝ) (hω : ω > 0) (α β : ℝ) 
  (h1 : f ω α = -1 / 2) (h2 : f ω β = 1 / 2) (h3 : abs (α - β) = 3 * Real.pi / 4) :
  ∃ (k : ℤ), ∀ (x : ℝ), -Real.pi / 2 + 3 * k * Real.pi ≤ x ∧ x ≤ Real.pi + 3 * k * Real.pi :=
by
  sorry

end interval_of_monotonic_increase_l411_411342


namespace sugar_content_of_mixture_l411_411712

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end sugar_content_of_mixture_l411_411712


namespace max_valid_license_plates_l411_411219

-- Definitions for the problem conditions
def isValidLicensePlate (a b : List ℕ) : Prop :=
  a.length = 6 ∧ b.length = 6 ∧ (∃ i j, i ≠ j ∧ (a.nth i) ≠ (b.nth i) ∧ (a.nth j) ≠ (b.nth j))

-- The theorem to prove the maximum number of such license plates
theorem max_valid_license_plates : ∃ S : Finset (List ℕ), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → isValidLicensePlate x y) ∧ S.card ≤ 100000 := sorry

end max_valid_license_plates_l411_411219


namespace rationalize_denominator_l411_411893

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l411_411893


namespace distinct_solutions_eq_three_l411_411909

theorem distinct_solutions_eq_three (x : ℝ) : 
  (|x - abs (3*x + 2)| = 4) → {x : ℝ | |x - abs (3*x + 2)| = 4}.finite ∧ {x : ℝ | |x - abs (3*x + 2)| = 4} = 3 :=
by
  -- proof skipped
  sorry

end distinct_solutions_eq_three_l411_411909


namespace james_eats_slices_l411_411839

theorem james_eats_slices :
  let initial_slices := 8 in
  let friend_eats := 2 in
  let remaining_slices := initial_slices - friend_eats in
  let james_eats := remaining_slices / 2 in
  james_eats = 3 := 
by 
  sorry

end james_eats_slices_l411_411839


namespace tangent_parallel_to_given_line_l411_411722

theorem tangent_parallel_to_given_line :
  let curve := λ x : ℝ, x^3 + x - 2
  let tangent_slope := λ x : ℝ, 3 * x^2 + 1
  let line_slope := 4
  let P1 := (1 : ℝ, 0)
  let P2 := (-1 : ℝ, -4)
  P1 ∈ {p : ℝ × ℝ | p.2 = curve p.1} ∧ tangent_slope P1.1 = line_slope ∧ 
  P2 ∈ {p : ℝ × ℝ | p.2 = curve p.1} ∧ tangent_slope P2.1 = line_slope :=
by
  let curve := λ x : ℝ, x^3 + x - 2
  let tangent_slope := λ x : ℝ, 3 * x^2 + 1
  let line_slope := 4
  let P1 := (1 : ℝ, 0)
  let P2 := (-1 : ℝ, -4)
  exact sorry

end tangent_parallel_to_given_line_l411_411722


namespace other_factor_one_l411_411449

theorem other_factor_one {p n : ℕ} (hp : Nat.Prime p) (hn : Nat.countDivisors n = 2) (hm : p ∣ n) : 
  ∃ d, (n = p * d ∧ d = 1) := 
by
  sorry

end other_factor_one_l411_411449


namespace ratio_of_differences_of_roots_l411_411352

theorem ratio_of_differences_of_roots 
  (a b : ℝ)
  (A B C D : ℝ)
  (hA : A = real.sqrt (a^2 + 12))
  (hB : B = real.sqrt (4 + 4*b))
  (hC : C = real.sqrt ((2 - 2*a)^2 + 12*(6 + b)) / 3)
  (hD : D = real.sqrt ((4 - a)^2 + 12*(3 + 2*b)) / 3)
  (hCD_neq : |C| ≠ |D|) : 
  A^2 - B^2 = 3 * (C^2 - D^2) := 
sorry

end ratio_of_differences_of_roots_l411_411352


namespace total_toys_l411_411364

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l411_411364


namespace artist_painting_rate_l411_411049

def mural_dimensions := (6 : ℕ, 3 : ℕ) -- 6m by 3m
def paint_cost_per_sqm := (4 : ℕ) -- $4 per square meter
def hourly_charge := (10 : ℕ) -- $10 per hour
def total_cost := (192 : ℕ) -- $192 total cost

theorem artist_painting_rate : 
  let area := (mural_dimensions.1 * mural_dimensions.2 : ℕ) in
  let paint_cost := (paint_cost_per_sqm * area : ℕ) in
  let labor_cost := (total_cost - paint_cost : ℕ) in
  let hours_worked := (labor_cost / hourly_charge : ℕ) in
  (area / hours_worked = 3 / 2) := 
by
  -- proof will go here
  sorry

end artist_painting_rate_l411_411049


namespace product_gcd_lcm_l411_411297

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l411_411297


namespace definite_integral_evaluation_l411_411715

theorem definite_integral_evaluation :
  ∫ x in 0..1, (sqrt(1 - (x - 1)^2) - 2 * x) = (Real.pi / 4) - 1 := by
  sorry

end definite_integral_evaluation_l411_411715


namespace rich_walks_x_road_distance_eq_200_l411_411880

-- Define the given problem and conditions
def rich_walks_distance (total_distance_walked: ℕ) (x_road_distance: ℕ) :=
  let distance_to_sidewalk := 20
  let distance_to_end_of_road := distance_to_sidewalk + x_road_distance
  let distance_after_left_turn := 2 * distance_to_end_of_road
  let total_distance_with_left := distance_to_sidewalk + x_road_distance + distance_after_left_turn
  let distance_half_again := (total_distance_with_left) / 2
  let distance_to_end_of_route := total_distance_with_left + distance_half_again
  2 * distance_to_end_of_route

-- Prove the distance to end of road x is 200 feet
theorem rich_walks_x_road_distance_eq_200 : 
  let x := 200 in
  rich_walks_distance 1980 x = 1980 :=
by
  sorry

end rich_walks_x_road_distance_eq_200_l411_411880


namespace z_factor_change_1_l411_411000

variable (q e x z : ℝ)
variable (k : ℝ) -- the factor by which z is changed

-- Conditions
def original_q : ℝ := (5 * e) / (4 * x * (z^2))

def new_q : ℝ := (5 * (4 * e)) / (4 * (2 * x) * ((k * z)^2))

-- Proof statement
theorem z_factor_change_1 (hq : q = (original_q q e x z))
                             (hq_new : (0.2222222222222222 * q) = (new_q q e x z k)) :
                             k = 1 :=
sorry

end z_factor_change_1_l411_411000


namespace max_triangles_from_parallel_lines_l411_411151

theorem max_triangles_from_parallel_lines (S1 S2 S3 : Finset (Set ℝ)) 
  (hS1 : S1.card = 10) (hS2 : S2.card = 10) (hS3 : S3.card = 10)
  (h_parallel_S1 : ∀ l1 l2 ∈ S1, Parallel l1 l2)
  (h_parallel_S2 : ∀ l1 l2 ∈ S2, Parallel l1 l2)
  (h_parallel_S3 : ∀ l1 l2 ∈ S3, Parallel l1 l2) :
  ∀ P : Set (Finset (Set ℝ)), (∀ l1 l2 l3 ∈ P, l1 ∉ S1 ∧ l2 ∉ S2 ∧ l3 ∉ S3 ∧ Intersect l1 l2 l3 ∧ P.card = 3) → P.card = 150 := 
sorry

end max_triangles_from_parallel_lines_l411_411151


namespace num_divisors_of_7p_l411_411448

open Nat

theorem num_divisors_of_7p (p : ℕ) (hp : Prime p) : ∃ d : ℕ, d = 4 ∧ n = 7 * p → ∃ n : ℕ, count_divisors n = 4 :=
sorry

end num_divisors_of_7p_l411_411448


namespace option_c_holds_l411_411027

noncomputable def k (n : ℝ) : ℝ := 2 * (n - 1)
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem option_c_holds (h : ∀ (n : ℝ), n ≠ 1) : 
  ∃ (n : ℝ), ∀ (x : ℝ), x > 0 → (inverse_proportion (k n - 1) x < 0) → 
  (∀ x1 x2 : ℝ, x1 < x2 → inverse_proportion (k n) x1 < inverse_proportion (k n) x2) :=
begin
  sorry -- The proof is omitted as per the instruction.
end

end option_c_holds_l411_411027


namespace proof_problem_l411_411029

-- Define the polynomial (1-x)(1+x)^4 and find the coefficient of x^2 term
def poly1 (x : ℝ) : ℝ := (1 - x) * (1 + x)^4

def coefficient_x2 : ℝ :=
  let term_with_x2 := (nat.choose 4 2) * (1 : ℝ)
  let term_with_x_x := (nat.choose 4 1) * (-1 : ℝ)
  term_with_x2 + term_with_x_x -- should equal 2

-- Define the polynomial (2 - 2x)^7 and find the specified sum of coefficients
def poly2 (x : ℝ) : ℝ := (2 - 2 * x)^7

def sum_of_even_odd_coefficients : ℝ :=
  let a0_to_a7 := array_of_terms poly2 7 -- placeholder function to get coefficients a_0 to a_7,
                                      -- this needs to be properly defined in Lean
  let sum_even := a0_to_a7[0] + a0_to_a7[2] + a0_to_a7[4] + a0_to_a7[6]
  let sum_odd := a0_to_a7[1] + a0_to_a7[3] + a0_to_a7[5] + a0_to_a7[7]
  sum_even^3 + sum_odd^3 -- should equal 0

theorem proof_problem :
  (coefficient_x2 = 2) ∧
  (sum_of_even_odd_coefficients = 0) :=
by
  sorry -- proof left out

end proof_problem_l411_411029


namespace suff_but_not_nec_condition_l411_411359

noncomputable def integral_result : ℝ := ∫ t in 1..2.718281828459045, 2 / t

theorem suff_but_not_nec_condition (x : ℝ) :
  x = integral_result → (∃ k : ℝ, (1, x) = k • (x, 4)) ∧ (¬∀ k : ℝ, (1, x) = k • (x, 4)) :=
sorry

end suff_but_not_nec_condition_l411_411359


namespace general_formula_sum_first_n_terms_l411_411323

open Nat

-- Definitions
def sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n+1)^2 = 2 * a n^2 + a n * a (n+1)

def specific_condition (a : ℕ → ℕ) :=
  a 2 + a 4 = 2 * a 3 + 4

-- Theorems (Proofs omitted with 'sorry')
theorem general_formula (a : ℕ → ℕ) (h1 : sequence a) (h2 : specific_condition a) : ∃ C : ℕ, ∀ n : ℕ, a n = C * 2^n :=
  sorry

theorem sum_first_n_terms (a : ℕ → ℕ) (h1 : sequence a) (h2 : specific_condition a) : 
  ∃ b : ℕ, ∀ n : ℕ, (finset.range n).sum (λ k, k * a k) = (n-1) * 2^(n+1) + 2 :=
  sorry

end general_formula_sum_first_n_terms_l411_411323


namespace base_of_parallelogram_l411_411721

-- Given conditions
variable (A : ℕ) (h : ℕ)
variable [hA : A = 288]
variable [hh : h = 16]

-- Prove that the base is 18
theorem base_of_parallelogram (b : ℕ) (hb : b = A / h) : b = 18 := by
  -- Placeholder for the proof
  sorry

end base_of_parallelogram_l411_411721


namespace probability_two_dice_show_1_l411_411579

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l411_411579


namespace magnification_factor_is_correct_l411_411242

theorem magnification_factor_is_correct
    (diameter_magnified_image : ℝ)
    (actual_diameter_tissue : ℝ)
    (diameter_magnified_image_eq : diameter_magnified_image = 2)
    (actual_diameter_tissue_eq : actual_diameter_tissue = 0.002) :
  diameter_magnified_image / actual_diameter_tissue = 1000 := by
  -- Theorem and goal statement
  sorry

end magnification_factor_is_correct_l411_411242


namespace ISBN_value_y_l411_411274

variables (A B C D E F G H I y J : ℕ)

-- Definitions based on provided conditions
def S : ℤ := 10 * 9 + 9 * 6 + 8 * 2 + 7 * y + 6 * 7 + 5 * 0 + 4 * 7 + 3 * 0 + 2 * 1
def r : ℤ := S % 11
def valid_J : ℤ :=
  if r = 0 then 0
  else if r = 1 then -1  -- representing 'x'
  else 11 - r

-- Now, the theorem to be proven
theorem ISBN_value_y :
  valid_J = 5 → y = 7 :=
begin
  -- Proof will go here
  sorry
end

end ISBN_value_y_l411_411274


namespace simplify_and_rationalize_l411_411890

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411890


namespace incorrect_propositions_l411_411687

theorem incorrect_propositions :
  (¬ (∀ x, (sin x = 1/2 → x = π/6)) ∧ 
  (¬ (∀ a b : ℝ, a < b → a^2 < b^2) ∧ ¬ (∀ a b : ℝ, a^2 < b^2 → a < b)) ∧
  (¬ (∀ x : ℝ, ¬(x^2 + x + 1 < 0)) ∧ ¬ (∀ x : ℝ, x^2 + x + 1 > 0))) :=
by
  sorry

end incorrect_propositions_l411_411687


namespace dice_probability_l411_411595

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411595


namespace dice_probability_l411_411596

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l411_411596


namespace locus_of_midpoint_l411_411846

noncomputable def midpoint (P Q : Point) : Point :=
⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

structure Circle :=
(center : Point)
(radius : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

theorem locus_of_midpoint
  (C1 C2 : Circle)
  (A1 : Point)
  (A2 : Point)
  (P1 : Point)
  (P2 : Point)
  (hA1_on_C1 : (A1.x - C1.center.x)^2 + (A1.y - C1.center.y)^2 = C1.radius^2)
  (hA2_on_C2 : (A2.x - C2.center.x)^2 + (A2.y - C2.center.y)^2 = C2.radius^2)
  (hP1_on_C1 : (P1.x - C1.center.x)^2 + (P1.y - C1.center.y)^2 = C1.radius^2)
  (hP2_on_C2 : (P2.x - C2.center.x)^2 + (P2.y - C2.center.y)^2 = C2.radius^2)
  (h_parallel : (A1.x - P1.x) * (A2.y - P2.y) = (A1.y - P1.y) * (A2.x - P2.x)) :
  ∃ O : Point, ∃ r : ℝ, ∀ M : Point, (M = midpoint P1 P2) → (M.x - O.x)^2 + (M.y - O.y)^2 = r^2 :=
sorry

end locus_of_midpoint_l411_411846


namespace q_0_plus_q_5_eq_225_l411_411437

noncomputable def q (ℝ → ℝ) := sorry

theorem q_0_plus_q_5_eq_225 (q : ℝ → ℝ) (h_monic : is_monic q) 
  (h_deg : degree q = 5)
  (h_q1 : q 1 = 21) 
  (h_q2 : q 2 = 42)
  (h_q3 : q 3 = 63)
  (h_q4 : q 4 = 84) : q 0 + q 5 = 225 :=
sorry

end q_0_plus_q_5_eq_225_l411_411437


namespace relay_team_orders_l411_411845

noncomputable def jordan_relay_orders : Nat :=
  let friends := [1, 2, 3] -- Differentiate friends; let's represent A by 1, B by 2, C by 3
  let choices_for_jordan_third := 2 -- Ways if Jordan runs third
  let choices_for_jordan_fourth := 2 -- Ways if Jordan runs fourth
  choices_for_jordan_third + choices_for_jordan_fourth

theorem relay_team_orders :
  jordan_relay_orders = 4 :=
by
  sorry

end relay_team_orders_l411_411845


namespace simplify_and_rationalize_l411_411899

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411899


namespace sequence_num_terms_l411_411511

theorem sequence_num_terms (n : ℕ) (h : n ≥ 2) : 
  ∃ k, k = n - 2 ∧ seq : Fin k → ℕ =λi, 2 * (i + 3) - 1 := sorry

end sequence_num_terms_l411_411511


namespace max_f_l411_411765

noncomputable def f (x : ℝ) : ℝ := real.sqrt x - real.log x

noncomputable def f' (x : ℝ) : ℝ := (1 / 2) * x^(-1 / 2) - 1 / x

theorem max_f' : ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f' x ≥ f' y) ∧ f' x = 1 / 16 :=
begin
  use 16,
  split,
  { norm_num, },
  split,
  { intro y,
    intro hy,
    -- Proof that f'(16) ≥ f'(y) omitted
    sorry,
  },
  { -- Proof that f'(16) = 1/16 omitted
    sorry,
  }
end

end max_f_l411_411765


namespace oblique_area_l411_411003

theorem oblique_area (side_length : ℝ) (A_ratio : ℝ) (S_original : ℝ) (S_oblique : ℝ) 
  (h1 : side_length = 1) 
  (h2 : A_ratio = (Real.sqrt 2) / 4) 
  (h3 : S_original = side_length ^ 2) 
  (h4 : S_oblique / S_original = A_ratio) : 
  S_oblique = (Real.sqrt 2) / 4 :=
by 
  sorry

end oblique_area_l411_411003


namespace circle_represents_range_l411_411493

theorem circle_represents_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 2 * y + 3 = 0 → (m > 2 * Real.sqrt 2 ∨ m < -2 * Real.sqrt 2)) :=
by
  sorry

end circle_represents_range_l411_411493


namespace average_rainfall_l411_411220

-- Define the conditions as individual definitions in Lean
def rain_first_30_minutes : ℝ := 5
def rain_next_30_minutes : ℝ := rain_first_30_minutes / 2
def rain_next_hour : ℝ := 1 / 2
def total_duration_hours : ℝ := 2

-- Sum total amount of rain
def total_rain : ℝ := rain_first_30_minutes + rain_next_30_minutes + rain_next_hour

-- Define the theorem to prove average rainfall per hour
theorem average_rainfall : total_rain / total_duration_hours = 4 := by
  sorry

end average_rainfall_l411_411220


namespace max_triangles_three_parallel_families_l411_411153

/-- 
Given three families of parallel lines, each containing 10 lines,
the maximum number of triangles that can be formed by these lines is 150.
-/
theorem max_triangles_three_parallel_families (L1 L2 L3 : set (set ℝ)) (h1 : ∀ l ∈ L1, ∀ l' ∈ L1, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y) 
  (h2 : ∀ l ∈ L2, ∀ l' ∈ L2, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y) 
  (h3 : ∀ l ∈ L3, ∀ l' ∈ L3, l ≠ l' → ∀ x ∈ l, ∀ y ∈ l', x ≠ y)
  (hL1_card : L1.card = 10) (hL2_card : L2.card = 10) (hL3_card : L3.card = 10) : 
  ∃ max_n : ℕ, max_n = 150 := 
begin
  sorry
end

end max_triangles_three_parallel_families_l411_411153


namespace probability_two_dice_showing_1_l411_411608

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l411_411608


namespace votes_for_E_and_F_l411_411397

theorem votes_for_E_and_F (total_votes : ℕ) (invalid_percentage : ℚ) 
  (valid_percentage_A valid_percentage_B valid_percentage_C valid_percentage_D : ℚ) :
  total_votes = 18000 → 
  invalid_percentage = 0.35 →
  valid_percentage_A = 0.32 →
  valid_percentage_B = 0.27 →
  valid_percentage_C = 0.18 →
  valid_percentage_D = 0.11 →
  let valid_votes := total_votes * (1 - invalid_percentage) in
  let remaining_percentage := 1 - (valid_percentage_A + valid_percentage_B + valid_percentage_C + valid_percentage_D) in
  let votes_E_and_F := valid_votes * remaining_percentage in
  votes_E_and_F = 1404 :=
begin
  sorry
end

end votes_for_E_and_F_l411_411397


namespace polyhedron_vertex_product_l411_411617

theorem polyhedron_vertex_product (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) 
    (label : ∀ {v₁ v₂ : V}, E v₁ v₂ → ℤ) (hV : Fintype.card V = 101) 
    (hE : ∀ {v₁ v₂ : V}, E v₁ v₂ → label (v₁:=v₁) (v₂:=v₂) = 1 ∨ label (v₁:=v₁) (v₂:=v₂) = -1)
    (hSymm : ∀ {v₁ v₂ : V} (h : E v₁ v₂), E v₂ v₁)
    (hEdges : ∀ v, ∃ v', E v v') :
  ∃ v : V, (∏ (w : V) in (Finset.filter (λ w, E v w) Finset.univ), label (v₁:=v) (v₂:=w)) = 1 :=
by
  sorry

end polyhedron_vertex_product_l411_411617


namespace monkey_climbing_time_l411_411214

theorem monkey_climbing_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) (net_gain : ℕ) :
  tree_height = 19 →
  hop_distance = 3 →
  slip_distance = 2 →
  net_gain = hop_distance - slip_distance →
  final_hop = hop_distance →
  (tree_height - final_hop) % net_gain = 0 →
  18 / net_gain + 1 = (tree_height - final_hop) / net_gain + 1 := 
by {
  sorry
}

end monkey_climbing_time_l411_411214


namespace total_students_l411_411391

theorem total_students (S F G B N : ℕ) 
  (hF : F = 41) 
  (hG : G = 22) 
  (hB : B = 9) 
  (hN : N = 24) 
  (h_total : S = (F + G - B) + N) : 
  S = 78 :=
by
  sorry

end total_students_l411_411391


namespace tangent_line_polar_l411_411037

theorem tangent_line_polar (ρ θ : ℝ) :
  let circle_eq := (ρ = 4 * sin θ)
  let point := (2 * sqrt 2, π / 4)
  (x^2 + y^2 - 4 * y = 0) → ∃ θ, ρ * cos θ = 2 :=
by
  sorry

end tangent_line_polar_l411_411037


namespace probability_two_ones_in_twelve_dice_l411_411583
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411583


namespace remainder_check_l411_411698

theorem remainder_check (q : ℕ) (n : ℕ) (h1 : q = 3^19) (h2 : n = 1162261460) : q % n = 7 := by
  rw [h1, h2]
  -- Proof skipped
  sorry

end remainder_check_l411_411698


namespace validate_propositions_l411_411240

theorem validate_propositions :
  (∀ l : Line, ¬ (l.is_vertical → l.has_inclination ∧ l.has_gradient)) ∧
  (¬ (∀ α, (0 ≤ α ∧ α ≤ Real.pi) → (inclination_range (l : Line) α ))) ∧
  (∀ α, ∀ l : Line, ¬ (l.gradient = Real.tan α → l.inclination = α)) ∧
  (∀ α, ∀ l : Line, ¬ (l.inclination = α → l.gradient = Real.tan α)) →
  count_correct_propositions = 0 :=
by {
  -- Definitions of the propositions
  sorry
}

end validate_propositions_l411_411240


namespace number_of_planks_needed_l411_411252

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l411_411252


namespace largest_intersection_value_l411_411123

theorem largest_intersection_value (c d b : ℝ) :
  ∃ x : ℝ, (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - b = c*x - d) ∧
          (∀ x₁ x₂ x₃ : ℝ, x₁, x₂, x₃ ≠ x → (x₁^5 - 5*x₁^4 + 10*x₁^3 - 10*x₁^2 + 5*x₁ - b = c*x₁ - d) →
                           (x₂^5 - 5*x₂^4 + 10*x₂^3 - 10*x₂^2 + 5*x₂ - b = c*x₂ - d) →
                           (x₃^5 - 5*x₃^4 + 10*x₃^3 - 10*x₃^2 + 5*x₃ - b = c*x₃ - d)) →
          (∃ x : ℝ, x = 1) :=
begin
  sorry
end

end largest_intersection_value_l411_411123


namespace extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l411_411122

theorem extremum_implies_derivative_zero {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_extremum : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :
  deriv f x₀ = 0 :=
sorry

theorem derivative_zero_not_implies_extremum {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_deriv_zero : deriv f x₀ = 0) :
  ¬ (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :=
sorry

end extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l411_411122


namespace simplify_and_rationalize_l411_411902

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411902


namespace painting_wall_l411_411783

theorem painting_wall :
  let heidi_rate := 1 / 60
  let linda_rate := 1 / 40
  let combined_rate := heidi_rate + linda_rate
  combined_rate * 12 = 1 / 2 :=
by
  let heidi_rate := 1 / 60
  let linda_rate := 1 / 40
  let combined_rate := heidi_rate + linda_rate
  have h : combined_rate = 1 / 24 := by sorry
  calc
    combined_rate * 12 = (1 / 24) * 12 : by rw [h]
                   ... = 1 / 2       : by sorry

end painting_wall_l411_411783


namespace area_of_unknown_square_l411_411730

theorem area_of_unknown_square
  (a b c d e : ℝ)
  (area_sq1 : ℝ) (area_sq2 : ℝ) (area_sq3 : ℝ)
  (h1 : area_sq1 = 3)
  (h2 : area_sq2 = 7)
  (h3 : area_sq3 = 22)
  (ha : a = area_sq1.sqrt)
  (hb : b = area_sq2.sqrt)
  (hc : c = area_sq3.sqrt)
  (right_triangle1 : d^2 = a^2 + c^2)
  (right_triangle2 : e^2 = b^2 + c^2) :
  d^2 - b^2 = 18 :=
begin
  -- proof goes here
  sorry
end

end area_of_unknown_square_l411_411730


namespace Tom_runs_60_miles_in_a_week_l411_411536

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l411_411536


namespace mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l411_411869

-- Definition of Mr. Wang's movements
def movements : List Int := [6, -3, 10, -8, 12, -7, -10]

-- Definitions of given conditions
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.3

theorem mr_wang_returned_to_1st_floor :
  (List.sum movements = 0) :=
by
  sorry

theorem mr_wang_electricity_consumption :
  (List.sum (movements.map Int.natAbs) * floor_height * electricity_per_meter = 50.4) :=
by
  sorry

end mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l411_411869


namespace unique_ids_div_ten_eq_312_l411_411647

-- Definitions reflecting the conditions in the given problem:
def valid_id_characters := {'A', 'I', 'M', 'E', '2', '0', '3'}
def allowed_count (ch : Char) : ℕ :=
  if ch = '2' ∨ ch = '3' then 2 else 1

def count_valid_ids : ℕ :=
  -- This would involve a detailed combinatorial calculation
  -- reflecting the steps performed in the provided solution.
  -- \(N\) is precomputed from the solution steps.
  3120

-- The theorem we need to prove:
theorem unique_ids_div_ten_eq_312 : count_valid_ids / 10 = 312 :=
by
  -- Placeholder for the proof: Implementing the actual combinatorial
  -- calculations is required to complete this theorem.
  sorry

end unique_ids_div_ten_eq_312_l411_411647


namespace problem_statement_l411_411035

-- Definitions used in the math problem
def number1 := 22 / 7
def number2 := 0.303003
def number3 := Real.sqrt 27
def number4 := Real.cbrt (-64)

-- Propositions to denote rationality and irrationality
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- The theorem statement
theorem problem_statement : 
  is_rational number1 ∧ 
  is_rational number2 ∧ 
  is_irrational number3 ∧ 
  is_rational number4 :=
by
  sorry

end problem_statement_l411_411035


namespace systematic_sampling_solution_l411_411813

noncomputable def systematic_sampling_problem : Prop :=
  ∀ (N M : ℕ),
  N = 1650 ∧ M = 35 →
  ∃ (R L : ℕ),
  R = N % M ∧ R = 5 ∧ L = M

theorem systematic_sampling_solution : systematic_sampling_problem :=
begin
  intros N M,
  assume h : N = 1650 ∧ M = 35,
  cases h with hN hM,
  use (N % M),
  split,
  { rw hN, rw hM, exact nat.mod_eq_of_lt (show 1650 % 35 = 5, from dec_trivial) },
  split,
  { rw hN, rw hM, exact nat.mod_eq_of_lt (show 1650 % 35 = 5, from dec_trivial) },
  { exact hM }
end

end systematic_sampling_solution_l411_411813


namespace base_of_log_is_176_l411_411334

theorem base_of_log_is_176 
    (x : ℕ)
    (h : ∃ q r : ℕ, x = 19 * q + r ∧ q = 9 ∧ r = 5) :
    x = 176 :=
by
  sorry

end base_of_log_is_176_l411_411334


namespace average_weight_of_whole_class_l411_411636

theorem average_weight_of_whole_class (n_a n_b : ℕ) (w_a w_b : ℕ) (avg_w_a avg_w_b : ℕ)
  (h_a : n_a = 36) (h_b : n_b = 24) (h_avg_a : avg_w_a = 30) (h_avg_b : avg_w_b = 30) :
  ((n_a * avg_w_a + n_b * avg_w_b) / (n_a + n_b) = 30) := 
by
  sorry

end average_weight_of_whole_class_l411_411636


namespace relationship_s_t_l411_411302

namespace TriangleProblem

variables {a b c : ℝ} (area circumradius : ℝ)
def s := sqrt a + sqrt b + sqrt c
def t := 1 / a + 1 / b + 1 / c

theorem relationship_s_t (h_area : area = 1/4) (h_circumradius : circumradius = 1) (h_abc : a * b * c = 1) : t > s :=
  sorry

end TriangleProblem

end relationship_s_t_l411_411302


namespace slope_of_tangent_at_0_l411_411513

theorem slope_of_tangent_at_0 (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (2 * x)) : 
  (deriv f 0) = 2 :=
sorry

end slope_of_tangent_at_0_l411_411513


namespace fraction_n_m_l411_411752

noncomputable def a (k : ℝ) := 2*k + 1
noncomputable def b (k : ℝ) := 3*k + 2
noncomputable def c (k : ℝ) := 3 - 4*k
noncomputable def S (k : ℝ) := a k + 2*(b k) + 3*(c k)

theorem fraction_n_m : 
  (∀ (k : ℝ), -1/2 ≤ k ∧ k ≤ 3/4 → (S (3/4) = 11 ∧ S (-1/2) = 16)) → 
  11/16 = 11 / 16 :=
by
  sorry

end fraction_n_m_l411_411752


namespace alice_original_seat_l411_411476

axiom initial_seating (alice ben cara dan ella fiona grace : ℕ) : Prop
axiom move_ben (ben : ℕ) : ben' = ben + 1
axiom move_cara (cara : ℕ) : cara' = cara - 2
axiom switch_dan_ella (dan ella : ℕ) : (dan', ella') = (ella, dan)
axiom move_fiona (fiona : ℕ) : fiona' = fiona + 3
axiom stay_grace (grace : ℕ) : grace' = grace
axiom alice_middle_seat (alice' : ℕ) : alice' = 4

theorem alice_original_seat (alice_start alice' : ℕ)
  (ben ben' cara cara' dan dan' ella ella' fiona fiona' grace grace' : ℕ)
  (h1 : move_ben ben = ben')
  (h2 : move_cara cara = cara')
  (h3 : switch_dan_ella dan ella = (dan', ella'))
  (h4 : move_fiona fiona = fiona')
  (h5 : stay_grace grace = grace')
  (h6 : alice_middle_seat alice' = 4)
  : alice_start = 6 := sorry

end alice_original_seat_l411_411476


namespace shelves_needed_l411_411462

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l411_411462


namespace simplify_and_rationalize_l411_411888

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l411_411888


namespace Leah_weeks_to_feed_l411_411426

structure Box :=
  (A B : ℕ)  -- A grams of Type A, B grams of Type B

structure Bird :=
  (weekly_total : ℕ)
  (A_ratio B_ratio : ℕ)

def pantry_boxes : List Box := [
  {A := 150, B := 100},
  {A := 125, B := 150},
  {A := 100, B := 125},
  {A := 200, B := 100},
  {A := 75, B := 200}
]

def parrot : Bird := {weekly_total := 100, A_ratio := 60, B_ratio := 40}
def cockatiel : Bird := {weekly_total := 50, A_ratio := 25, B_ratio := 25}
def canary : Bird := {weekly_total := 25, A_ratio := 10, B_ratio := 15}

def weekly_consumption (bird : Bird) : Box :=
  { A := bird.weekly_total * bird.A_ratio / 100, B := bird.weekly_total * bird.B_ratio / 100 }

def total_weekly_consumption : Box :=
  { A := weekly_consumption parrot .A + weekly_consumption cockatiel .A + weekly_consumption canary .A,
    B := weekly_consumption parrot .B + weekly_consumption cockatiel .B + weekly_consumption canary .B }

def total_seed_in_pantry : Box :=
  { A := pantry_boxes.map Box.A |>.sum, B := pantry_boxes.map Box.B |>.sum }

noncomputable def weeks_to_feed : ℕ :=
  min (total_seed_in_pantry.A / total_weekly_consumption.A) 
      (total_seed_in_pantry.B / total_weekly_consumption.B)

theorem Leah_weeks_to_feed : weeks_to_feed = 6 := by
  sorry

end Leah_weeks_to_feed_l411_411426


namespace fraction_of_remaining_paint_used_l411_411844

theorem fraction_of_remaining_paint_used (total_paint : ℕ) (first_week_fraction : ℚ) (total_used : ℕ) :
  total_paint = 360 ∧ first_week_fraction = 1/6 ∧ total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
  by
    sorry

end fraction_of_remaining_paint_used_l411_411844


namespace servings_needed_l411_411146

theorem servings_needed
  (pieces_per_serving : ℕ)
  (jared_consumption : ℕ)
  (three_friends_consumption : ℕ)
  (another_three_friends_consumption : ℕ)
  (last_four_friends_consumption : ℕ) : 
  pieces_per_serving = 60 →
  jared_consumption = 150 →
  three_friends_consumption = 3 * 80 →
  another_three_friends_consumption = 3 * 200 →
  last_four_friends_consumption = 4 * 100 →
  ∃ (s : ℕ), s = 24 :=
by
  intros
  sorry

end servings_needed_l411_411146


namespace probability_exactly_two_ones_l411_411555

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411555


namespace perpendicular_planes_l411_411356

-- Definitions of lines and planes
variables (a b : Line) (α β : Plane)

-- Hypotheses
hypothesis (h1 : a ≠ b)
hypothesis (h2 : α ≠ β)
hypothesis (h3 : a ⊥ b)
hypothesis (h4 : a ⊥ α)
hypothesis (h5 : b ⊥ β)

-- Theorem statement
theorem perpendicular_planes (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : a ⊥ b) 
  (h4 : a ⊥ α) 
  (h5 : b ⊥ β) : 
  α ⊥ β :=
sorry

end perpendicular_planes_l411_411356


namespace add_base3_numbers_l411_411231

-- Definitions to represent the numbers in base 3
def base3_num1 := (2 : ℕ) -- 2_3
def base3_num2 := (2 * 3 + 2 : ℕ) -- 22_3
def base3_num3 := (2 * 3^2 + 0 * 3 + 2 : ℕ) -- 202_3
def base3_num4 := (2 * 3^3 + 0 * 3^2 + 2 * 3 + 2 : ℕ) -- 2022_3

-- Summing the numbers in base 10 first
def sum_base10 := base3_num1 + base3_num2 + base3_num3 + base3_num4

-- Expected result in base 10 for 21010_3
def result_base10 := 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3 + 0

-- Proof statement
theorem add_base3_numbers : sum_base10 = result_base10 :=
by {
  -- Proof not required, so we skip it using sorry
  sorry
}

end add_base3_numbers_l411_411231


namespace simplify_and_rationalize_l411_411901

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l411_411901


namespace trader_gain_cost_of_pens_l411_411697

-- Definitions based on the conditions
variables (C S : ℝ) (number_of_pens_sold : ℝ) (gain_percentage : ℝ)

-- Given conditions
def conditions := number_of_pens_sold = 95 ∧ gain_percentage = 20

-- The main theorem to prove
theorem trader_gain_cost_of_pens (h : conditions C S number_of_pens_sold gain_percentage) : 
  ∃ N, N * C = number_of_pens_sold * (C * (1 + gain_percentage / 100)) - number_of_pens_sold * C ∧ N = 19 :=
by 
  sorry  -- Proof to be filled in later

end trader_gain_cost_of_pens_l411_411697


namespace trig_expression_zero_l411_411757

theorem trig_expression_zero (α : ℝ) (h : Real.tan α = 2) : 
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := 
by
  sorry

end trig_expression_zero_l411_411757


namespace min_value_neg_inf_l411_411331

-- Definitions
variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

variables (a b : ℝ)
variables (f g : ℝ → ℝ) [is_odd f] [is_odd g]
variable F : ℝ → ℝ := λ x, a * f x + b * g x + 2

-- Given conditions as assumptions
axiom ab_ne_zero : a ≠ 0 ∧ b ≠ 0
axiom max_val_pos : ∃ x : ℝ, x > 0 ∧ F x = 5

-- Prove the minimum value of F over (-∞, 0) is -1.
theorem min_value_neg_inf : ∃ x : ℝ, x < 0 ∧ F x = -1 :=
sorry

end min_value_neg_inf_l411_411331


namespace max_sum_value_l411_411055

theorem max_sum_value {n : ℕ} (h_pos : 0 < n) (x : Fin n → ℝ) (h_bounds : ∀ i, -1 ≤ x i ∧ x i ≤ 1) :
  (∑ r in Finset.range (2*n), ∑ s in Finset.range (2*n), if r < s then (s-r-n)*x r*x s else 0) ≤ n*(n-1) :=
sorry

end max_sum_value_l411_411055


namespace place_125_is_41000_l411_411236

-- Conditions: Natural numbers, sum of digits equals 5, arranged in ascending order
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

def is_valid_number (n : ℕ) : Prop :=
  sum_of_digits n = 5

def valid_numbers : list ℕ :=
  list.sort (≤) (list.filter is_valid_number (list.range 100000)) -- Considering 5-digit numbers as upper bound

theorem place_125_is_41000 : valid_numbers.nth 124 = some 41000 :=
by
  -- Proof goes here
  sorry

end place_125_is_41000_l411_411236


namespace place_125_is_41000_l411_411235

-- Conditions: Natural numbers, sum of digits equals 5, arranged in ascending order
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

def is_valid_number (n : ℕ) : Prop :=
  sum_of_digits n = 5

def valid_numbers : list ℕ :=
  list.sort (≤) (list.filter is_valid_number (list.range 100000)) -- Considering 5-digit numbers as upper bound

theorem place_125_is_41000 : valid_numbers.nth 124 = some 41000 :=
by
  -- Proof goes here
  sorry

end place_125_is_41000_l411_411235


namespace monotonic_decreasing_intervals_l411_411772

open Real

def f (x : ℝ) : ℝ :=
  sin (2 * x) * tan (x) + cos (2 * x - π / 3) - 1

theorem monotonic_decreasing_intervals (k : ℤ) :
  is_monotonic_decreasing_on f 
    (Ico (k * π + π / 3) (k * π + π / 2)) ∧
    is_monotonic_decreasing_on f 
    (Ioc (k * π + π / 2) (k * π + 5 * π / 6)) := sorry

end monotonic_decreasing_intervals_l411_411772


namespace largest_sequence_exists_l411_411070

theorem largest_sequence_exists :
  let house_number := [9, 0, 2, 3, 4]
  let house_number_sum := 9 + 0 + 2 + 3 + 4
  ∃ (office_phone_number : List ℕ), 
    office_phone_number.length = 10 ∧ 
    (office_phone_number.sum = house_number_sum) ∧ 
    LexicographicalMax office_phone_number [9, 0, 5, 4, 0, 0, 0, 0, 0] := by
  sorry

end largest_sequence_exists_l411_411070


namespace sequence_general_term_l411_411919

-- Define the sequence using a recurrence relation for clarity in formal proof
def a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n + 1 else 3

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → a n = 2^n + 1 := 
by 
  sorry

end sequence_general_term_l411_411919


namespace retirement_savings_l411_411081

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l411_411081


namespace sum_of_g_diverges_l411_411731

namespace SumOfSeries

open Real

noncomputable def g (n : ℕ) : ℝ := ∑' k, 1 / (k ^ n)

theorem sum_of_g_diverges : ∑' n, g n = ⊤ :=
by
  sorry

end SumOfSeries

end sum_of_g_diverges_l411_411731


namespace probability_two_ones_in_twelve_dice_l411_411582
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411582


namespace cookies_left_l411_411423

def total_cookies_bought : ℕ := 24
def percentage_eaten : ℝ := 0.25
def cookies_eaten (total: ℕ) (percentage: ℝ) : ℕ := (percentage * total).toNat

theorem cookies_left (total: ℕ) (percentage: ℝ) : 
  percentage = 0.25 ∧ total = 24 → total - cookies_eaten total percentage = 18 := 
by
  intros h
  cases h with h_pct h_total
  rw [h_total, h_pct]
  simp [cookies_eaten]
  norm_num
  sorry

end cookies_left_l411_411423


namespace value_of_expression_l411_411672

theorem value_of_expression : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end value_of_expression_l411_411672


namespace part1_part2_l411_411931

open ArithmeticSequence

variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}
variables (n : ℕ)

-- Conditions
axiom seq_a_arithmetic : arithmetic_sequence a
axiom seq_b_arithmetic : arithmetic_sequence b
axiom sum_of_n_terms : S n = sum_first_n_terms a n
axiom sum_of_n_terms_b : T n = sum_first_n_terms b n
axiom given_ratio : ∀ n, S n / T n = (3 * n + 1) / (n + 3)

-- Prove part 1
theorem part1 : (a 2 + a 20) / (b 7 + b 15) = 8 / 3 := 
    sorry

-- Prove part 2
theorem part2 : {n : ℕ | (a n / b n).is_integer}.card = 2 := 
    sorry

end part1_part2_l411_411931


namespace hexagon_angle_equality_l411_411817

variables (A B C D E F : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] 
          [decidable_eq D] [decidable_eq E] [decidable_eq F]

-- Angles referenced in hexagon
variables (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ)

-- Conditions: Equilateral hexagon and given angle sum equality
variables (h1 : angle_A + angle_C + angle_E = angle_B + angle_D + angle_F)

-- We need to prove the following statements
theorem hexagon_angle_equality 
  (h2 : equilateral_hexagon A B C D E F) 
: angle_A = angle_D ∧ angle_B = angle_E ∧ angle_C = angle_F := 
by
  sorry

end hexagon_angle_equality_l411_411817


namespace pentagram_intersections_l411_411748

structure Pentagram :=
(vertices : Fin 5 → Point)
(no_three_collinear : ∀ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (vertices i) (vertices j) (vertices k))
(self_intersections : Set Point)

def possible_self_intersections (p : Pentagram) : Prop :=
  p.self_intersections = 1 ∨ p.self_intersections = 2 ∨ p.self_intersections = 3 ∨ p.self_intersections = 5

theorem pentagram_intersections (p : Pentagram) :
  possible_self_intersections p ∧ ¬(p.self_intersections = 4) :=
by
  sorry

end pentagram_intersections_l411_411748


namespace last_two_digits_of_sum_of_first_50_factorials_l411_411964

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l411_411964


namespace computer_hardware_contract_prob_l411_411135

theorem computer_hardware_contract_prob :
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  ∃ P_H : ℝ, P_at_least_one = P_H + P_S - P_H_and_S ∧ P_H = 0.8 :=
by
  -- Let definitions and initial conditions
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  -- Solve for P(H)
  let P_H := 0.8
  -- Show the proof of the calculation
  sorry

end computer_hardware_contract_prob_l411_411135


namespace michael_scoops_l411_411865

def needs_flour_cups : ℝ := 6
def total_flour_cups : ℝ := 8
def measuring_cup_size : ℝ := 1 / 4

theorem michael_scoops : (total_flour_cups - needs_flour_cups) / measuring_cup_size = 8 := 
  by
  sorry

end michael_scoops_l411_411865


namespace find_original_population_l411_411680

-- Defining the conditions
def original_population (P : ℕ) := 
  let new_population := P + 100 in
  let after_moving_out_population := new_population - 400 in
  let population_after_4_years := after_moving_out_population / (2 ^ 4) in
  population_after_4_years = 60

-- The theorem stating the original population
theorem find_original_population : ∃ P : ℕ, original_population P ∧ P = 1260 :=
by
  sorry

end find_original_population_l411_411680


namespace probability_exactly_two_ones_l411_411557

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l411_411557


namespace question_mark_value_l411_411995

theorem question_mark_value (x : ℝ) (h : (sqrt x) / 18 = 4) : x = 5184 :=
sorry

end question_mark_value_l411_411995


namespace combined_weight_of_boxes_l411_411457

-- Variables for the weights of the four boxes
variables (a b c d : ℕ)

-- Given conditions
def condition_1 : Prop := a + b + c = 135
def condition_2 : Prop := a + b + d = 139
def condition_3 : Prop := a + c + d = 142
def condition_4 : Prop := b + c + d = 145

-- The main theorem to prove
theorem combined_weight_of_boxes (h1 : condition_1 a b c) 
                                 (h2 : condition_2 a b d) 
                                 (h3 : condition_3 a c d) 
                                 (h4 : condition_4 b c d) : 
                                 a + b + c + d = 187 :=
begin
  -- This is where the proof would go
  sorry
end

end combined_weight_of_boxes_l411_411457


namespace polygon_sides_l411_411836

theorem polygon_sides (h₁ : ∀ (a d : ℝ) (n : ℕ), 
                        a = 120 ∧ d = 5 → 
                        let angles := list.range n |>.map (λ i, a + i * d) in 
                        n * (2 * a + (n - 1) * d) / 2 = 180 * (n - 2) ∧ 
                        ∀ ⦃i⦄, i ∈ list.range n → angles.nth i < 180) : 
                        n = 9 :=
by 
have h := h₁ 120 5 n
sorry

end polygon_sides_l411_411836


namespace problem_statement_l411_411651

/-- Define the function f on positive integers -/
def f : ℕ → ℕ
| 1            := 1
| (2 * n)      := f n
| (4 * n + 1)  := 2 * f (2 * n + 1) - f n
| (4 * n + 3)  := 3 * f (2 * n + 1) - 2 * f n 
| _            := 0 -- default case for non-covered inputs, e.g., 0

/-- Define the property we want to prove -/
def number_of_solutions (bound : ℕ) : ℕ :=
  Finset.card { n ∈ Finset.range (bound + 1) | f n = n }

/-- The statement of the problem -/
theorem problem_statement : number_of_solutions 1988 = 92 := sorry

end problem_statement_l411_411651


namespace bases_final_digit_625_is_1_l411_411306

theorem bases_final_digit_625_is_1 : 
  {b : ℕ | 2 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0}.card = 5 := 
by
  sorry

end bases_final_digit_625_is_1_l411_411306


namespace lisa_balls_count_l411_411103

def stepNumber := 1729

def base7DigitsSum(x : Nat) : Nat :=
  x / 7 ^ 3 + (x % 343) / 7 ^ 2 + (x % 49) / 7 + x % 7

theorem lisa_balls_count (h1 : stepNumber = 1729) : base7DigitsSum stepNumber = 11 := by
  sorry

end lisa_balls_count_l411_411103


namespace semicircle_area_correct_l411_411927

-- Let r be the radius of the semicircle
def r := 3

-- The given perimeter of the semicircle
def semicirclePerimeter : ℝ := 15.42

-- Derived condition for the radius
def derivedRadius : ℝ := semicirclePerimeter / 5.14

-- Given the perimeter calculation
axiom perimeter_eq (r : ℝ) : 3.14 * r + 2 * r = semicirclePerimeter

-- Calculating the area of the semicircle
def semicircleArea (r : ℝ) : ℝ := 3.14 * (r ^ 2) / 2

-- Problem statement: Prove that the area of the semicircle matches the given area.
theorem semicircle_area_correct :
  semicircleArea r = 14.13 := by
  sorry

end semicircle_area_correct_l411_411927


namespace average_rainfall_l411_411222

-- We define the given conditions as separate definitions.
def first_30_min_rain : ℝ := 5
def next_30_min_rain : ℝ := first_30_min_rain / 2
def next_hour_rain : ℝ := 0.5

-- The total rainfall over the duration of the storm.
def total_rainfall : ℝ := first_30_min_rain + next_30_min_rain + next_hour_rain

-- The total duration of the storm in hours.
def total_duration : ℝ := 2 -- 30 minutes + 30 minutes + 1 hour

-- We now write the theorem statement to prove that the average rainfall total is 4 inches per hour.
theorem average_rainfall : (total_rainfall / total_duration) = 4 := by
  -- All the steps for the proof would go here.
  sorry

end average_rainfall_l411_411222


namespace probability_two_ones_in_twelve_dice_l411_411584
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l411_411584


namespace original_daily_production_l411_411650

theorem original_daily_production (x N : ℕ) (h1 : N = (x - 3) * 31 + 60) (h2 : N = (x + 3) * 25 - 60) : x = 8 :=
sorry

end original_daily_production_l411_411650


namespace coin_probability_l411_411987

theorem coin_probability :
  (∑ outcome in {ω | (∃ (i : Fin 3), ω i = tt) ∧ (∀ (i : Fin 3), ω i = tt)}, 1 / 2 ^ 3) /
  (∑ outcome in {ω | ∃ (i : Fin 3), ω i = tt}, 1 / 2 ^ 3) = 1 / 7 :=
by
  sorry

end coin_probability_l411_411987


namespace total_squares_after_removals_l411_411950

/-- 
Prove that the total number of squares of various sizes on a 5x5 grid,
after removing two 1x1 squares, is 55.
-/
theorem total_squares_after_removals (total_squares_in_5x5_grid: ℕ) (removed_squares: ℕ) : 
  (total_squares_in_5x5_grid = 25 + 16 + 9 + 4 + 1) →
  (removed_squares = 2) →
  (total_squares_in_5x5_grid - removed_squares = 55) :=
sorry

end total_squares_after_removals_l411_411950


namespace factorial_sum_power_of_two_l411_411281

theorem factorial_sum_power_of_two (a b c n : ℕ) (h : a ≤ b ∧ b ≤ c) :
  a! + b! + c! = 2^n →
  (a = 1 ∧ b = 1 ∧ c = 2) ∨
  (a = 1 ∧ b = 1 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 4) ∨
  (a = 2 ∧ b = 3 ∧ c = 5) :=
by
  sorry

end factorial_sum_power_of_two_l411_411281


namespace equilateral_triangle_perimeter_and_area_l411_411403

theorem equilateral_triangle_perimeter_and_area (s : ℝ) (h_s : s = 10) :
  let P := 3 * s in
  let A := (sqrt 3 / 4) * s^2 in
  P = 30 ∧ A = 25 * sqrt 3 :=
by
  have hP : P = 3 * s := rfl
  have hA : A = (sqrt 3 / 4) * s^2 := rfl
  rw [h_s, hP, hA]
  split
  { norm_num }
  { norm_num, ring, sorry } -- using sorry to skip the detailed proof

end equilateral_triangle_perimeter_and_area_l411_411403
