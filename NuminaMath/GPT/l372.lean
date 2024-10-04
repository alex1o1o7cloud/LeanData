import Mathlib

namespace typing_lines_in_10_minutes_l372_372889

def programmers := 10
def total_lines_in_60_minutes := 60
def total_minutes := 60
def target_minutes := 10

theorem typing_lines_in_10_minutes :
  (total_lines_in_60_minutes / total_minutes) * programmers * target_minutes = 100 :=
by sorry

end typing_lines_in_10_minutes_l372_372889


namespace find_a_l372_372064

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372064


namespace negation_of_universal_proposition_l372_372120

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_proposition_l372_372120


namespace task_completion_choice_l372_372226

theorem task_completion_choice (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 := by
  sorry

end task_completion_choice_l372_372226


namespace total_yearly_car_leasing_cost_l372_372528

-- Define mileage per day
def mileage_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" ∨ day = "Sunday" then 50
  else if day = "Tuesday" ∨ day = "Thursday" then 80
  else if day = "Saturday" then 120
  else 0

-- Define weekly mileage
def weekly_mileage : ℕ := 4 * 50 + 2 * 80 + 120

-- Define cost parameters
def cost_per_mile : ℕ := 1 / 10
def weekly_fee : ℕ := 100
def monthly_toll_parking_fees : ℕ := 50
def discount_every_5th_week : ℕ := 30
def number_of_weeks_in_year : ℕ := 52

-- Define total yearly cost
def total_cost_yearly : ℕ :=
  let total_weekly_cost := (weekly_mileage * cost_per_mile + weekly_fee)
  let total_yearly_cost := total_weekly_cost * number_of_weeks_in_year
  let total_discounts := (number_of_weeks_in_year / 5) * discount_every_5th_week
  let annual_cost_without_tolls := total_yearly_cost - total_discounts
  let total_toll_fees := monthly_toll_parking_fees * 12
  annual_cost_without_tolls + total_toll_fees

-- Define the main theorem
theorem total_yearly_car_leasing_cost : total_cost_yearly = 7996 := 
  by
    -- Proof omitted
    sorry

end total_yearly_car_leasing_cost_l372_372528


namespace amount_per_person_l372_372917

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each_person : ℕ) :
  total_amount = 42900 → num_persons = 22 → amount_each_person = 1950 →
  total_amount / num_persons = amount_each_person :=
by 
  intros h_total h_num h_each
  rw [h_total, h_num, h_each]
  exact nat.div_self h_each -- this needs to be adjusted to fit actual Lean library usage
  sorry -- Placeholder for the actual proof

end amount_per_person_l372_372917


namespace floor_inequality_sqrt_l372_372840

theorem floor_inequality_sqrt (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (⌊ m * Real.sqrt 2 ⌋) * (⌊ n * Real.sqrt 7 ⌋) < (⌊ m * n * Real.sqrt 14 ⌋) := 
by
  sorry

end floor_inequality_sqrt_l372_372840


namespace part1_part2_l372_372994

def nearest_integer (x : ℝ) : ℕ :=
  if x - x.floor < 0.5 then x.floor.nat_abs else (x.floor + 1).nat_abs

def G (x : ℝ) : ℕ := nearest_integer x

axiom G_conditions : G (4 / 3) = 1 ∧ G (5 / 3) = 2 ∧ G 2 = 2 ∧ G (2.5) = 3

theorem part1 : 
  G_conditions ∧ 
  G 1 = 1 ∧ 
  G 3 = 3 ∧ 
  G 4 = 4 → 
  (1 / G 1) + (1 / G 2) + (1 / G 3) + (1 / G 4) = 25 / 12 :=
sorry

theorem part2 : 
  G_conditions ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2022 → 1 / G (sqrt n)) → 
  ∑ k in (finset.range 2022), 1 / (G (sqrt (k + 1))) = 1334 / 15 :=
sorry

end part1_part2_l372_372994


namespace truck_capacity_l372_372575

theorem truck_capacity :
  ∀ (max_load bag_mass num_bags : ℕ), 
    max_load = 900 → bag_mass = 8 → num_bags = 100 → 
    max_load - (bag_mass * num_bags) = 100 :=
by
  intros max_load bag_mass num_bags Hmax Hbag Hnum
  rw [Hmax, Hbag, Hnum]
  exact rfl

end truck_capacity_l372_372575


namespace cos_A_proof_area_of_triangle_proof_l372_372382

noncomputable def cos_A (a b c : ℝ) (h : c * cos A + a * cos C = 2 * b * cos A) : ℝ :=
  if h : c + a ≠ 2 * b then sorry else 1 / 2  -- condition directly from the problem

noncomputable def area_of_triangle (a b c : ℝ) (h1 : a = sqrt 7) (h2 : b + c = 4) (h3 : cos_A a b c _ = 1 / 2) : ℝ :=
  if h : b * c ≠ 0 then sorry
  else 3 * sqrt 3 / 4 -- condition directly from the problem

-- Proof statements
theorem cos_A_proof (a b c : ℝ) (h : c * cos A + a * cos C = 2 * b * cos A) : cos_A a b c h = 1 / 2 :=
  by sorry

theorem area_of_triangle_proof (a b c : ℝ) (h1 : a = sqrt 7) (h2 : b + c = 4) (h3 : cos_A a b c _) : 
    area_of_triangle a b c h1 h2 h3 = 3 * sqrt 3 / 4 :=
  by sorry

end cos_A_proof_area_of_triangle_proof_l372_372382


namespace midpoint_of_AQ_l372_372774

theorem midpoint_of_AQ 
  {A B C P Q M : Type*} [VectorSpace ℝ (V := Type*) ]
  (hP : ∃ t : ℝ, ∀ (CA CB : V), t • CA + (1 - t) • CB = (2 / 3: ℝ) • CA + (1 / 3: ℝ) • CB ∧ P ∈ AB)
  (hQ : Q ∈ Segment B C ∧ ∀(B C : V), Q = (1 / 2) • B + (1 / 2) • C)
  (hM : ∃ t : ℝ, ∀ (AM AQ : V), M ∈ Segment A Q ∧ t • CP = CM) :
  ∃ λ : ℝ, λ = (1 / 2) ∧ ∀ (AQ : V), M = (1 / 2) • A + (1 / 2) • Q := 
sorry

end midpoint_of_AQ_l372_372774


namespace positive_t_solution_l372_372294

noncomputable def positive_t : ℝ :=
  classical.some (exists_unique 
    (λ t : ℝ, t > 0 ∧ ∃ a b : ℂ, |a| = 3 ∧ |b| = 5 ∧ ab = t - 3 * complex.I))

theorem positive_t_solution :
  ∃ t : ℝ, t > 0 ∧ ∃ a b : ℂ, |a| = 3 ∧ |b| = 5 ∧ ab = t - 3 * complex.I :=
  sorry

example : positive_t = 6 * real.sqrt 6 :=
  sorry

end positive_t_solution_l372_372294


namespace finite_obtuse_extendable_infinite_obtuse_not_extendable_l372_372915

-- Definitions for non-degenerate obtuse triangle
def is_obtuse_triangle (P Q R : Point) : Prop := 
  -- definition of a non-degenerate obtuse triangle

def satisfies_obtuse_property (S : Set Point) : Prop :=
  ∀ P Q R ∈ S, is_obtuse_triangle P Q R

-- (a) Statement of the finite set problem
theorem finite_obtuse_extendable (S : Set Point) (h_finite : S.finite) (h_obtuse : satisfies_obtuse_property S) :
  ∃ T : Set Point, S ⊆ T ∧ T.finite ∧ satisfies_obtuse_property T :=
sorry

-- (b) Statement of the infinite set problem
theorem infinite_obtuse_not_extendable :
  ∃ S : Set Point, S.infinite ∧ satisfies_obtuse_property S ∧ ¬∃ T : Set Point, S ⊆ T ∧ satisfies_obtuse_property T :=
sorry

end finite_obtuse_extendable_infinite_obtuse_not_extendable_l372_372915


namespace interval_necessary_not_sufficient_l372_372924

theorem interval_necessary_not_sufficient :
  (∀ x, x^2 - x - 2 = 0 → (-1 ≤ x ∧ x ≤ 2)) ∧ (∃ x, x^2 - x - 2 = 0 ∧ ¬(-1 ≤ x ∧ x ≤ 2)) → False :=
by
  sorry

end interval_necessary_not_sufficient_l372_372924


namespace find_number_l372_372913

theorem find_number 
  (x : ℝ) 
  (h1 : 3 * (2 * x + 9) = 69) : x = 7 := by
  sorry

end find_number_l372_372913


namespace minimum_red_points_for_square_l372_372522

/-- Given a circle divided into 100 equal segments with points randomly colored red. 
Prove that the minimum number of red points needed to ensure at least four red points 
form the vertices of a square is 76. --/
theorem minimum_red_points_for_square (n : ℕ) (h : n = 100) (red_points : Finset ℕ)
  (hred : red_points.card ≥ 76) (hseg : ∀ i j : ℕ, i ≤ j → (j - i) % 25 ≠ 0 → ¬ (∃ a b c d : ℕ, 
  a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0)) : 
  ∃ a b c d : ℕ, a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0 :=
sorry

end minimum_red_points_for_square_l372_372522


namespace weight_of_currants_l372_372593

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end weight_of_currants_l372_372593


namespace proof_problem_l372_372443

variables {x y z w : ℝ}

-- Condition given in the problem
def condition (x y z w : ℝ) : Prop :=
  (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3

-- The statement to be proven
theorem proof_problem (h : condition x y z w) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 :=
by
  sorry

end proof_problem_l372_372443


namespace probability_no_defective_pencils_l372_372185

theorem probability_no_defective_pencils :
  let total_pencils := 9
  let defective_pencils := 2
  let total_ways_choose_3 := Nat.choose total_pencils 3
  let non_defective_pencils := total_pencils - defective_pencils
  let ways_choose_3_non_defective := Nat.choose non_defective_pencils 3
  (ways_choose_3_non_defective : ℚ) / total_ways_choose_3 = 5 / 12 :=
by
  sorry

end probability_no_defective_pencils_l372_372185


namespace popsicle_sticks_sum_l372_372562

-- Define the number of popsicle sticks each person has
def Gino_popsicle_sticks : Nat := 63
def my_popsicle_sticks : Nat := 50

-- Formulate the theorem stating the sum of popsicle sticks
theorem popsicle_sticks_sum : Gino_popsicle_sticks + my_popsicle_sticks = 113 := by
  sorry

end popsicle_sticks_sum_l372_372562


namespace solve_quadratic_eq_l372_372096

-- Define the equation as part of the conditions
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 + 4 * x - 1 = 0

theorem solve_quadratic_eq (x : ℝ) :
  quadratic_eq x ↔ (x = -1 + sqrt 6 / 2 ∨ x = -1 - sqrt 6 / 2) :=
by sorry

end solve_quadratic_eq_l372_372096


namespace number_of_pieces_of_string_l372_372181

theorem number_of_pieces_of_string (total_length piece_length : ℝ) (h1 : total_length = 60) (h2 : piece_length = 0.6) :
    total_length / piece_length = 100 := by
  sorry

end number_of_pieces_of_string_l372_372181


namespace product_of_solutions_product_of_all_t_l372_372673

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372673


namespace skyler_total_songs_skyler_success_breakdown_l372_372850

noncomputable def skyler_songs : ℕ :=
  let hit_songs := 25
  let top_100_songs := hit_songs + 10
  let unreleased_songs := hit_songs - 5
  let duets_total := 12
  let duets_top_20 := duets_total / 2
  let duets_not_top_200 := duets_total / 2
  let soundtracks_total := 18
  let soundtracks_extremely := 3
  let soundtracks_moderate := 8
  let soundtracks_lukewarm := 7
  let projects_total := 22
  let projects_global := 1
  let projects_regional := 7
  let projects_overlooked := 14
  hit_songs + top_100_songs + unreleased_songs + duets_total + soundtracks_total + projects_total

theorem skyler_total_songs : skyler_songs = 132 := by
  sorry

theorem skyler_success_breakdown :
  let extremely_successful := 25 + 1
  let successful := 35 + 6 + 3
  let moderately_successful := 8 + 7
  let less_successful := 7 + 14 + 6
  let unreleased := 20
  (extremely_successful, successful, moderately_successful, less_successful, unreleased) =
  (26, 44, 15, 27, 20) := by
  sorry

end skyler_total_songs_skyler_success_breakdown_l372_372850


namespace abs_z2_minus_2z_eq_2_l372_372749

theorem abs_z2_minus_2z_eq_2 (z : ℂ) (h : z = 1 + complex.I) : abs (z^2 - 2*z) = 2 := by
  sorry

end abs_z2_minus_2z_eq_2_l372_372749


namespace birds_on_the_fence_l372_372861

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l372_372861


namespace number_of_ideal_subsets_l372_372426

/-- An ideal subset of natural numbers -/
def is_ideal (p q : ℕ) (S : set ℕ) : Prop :=
  0 ∈ S ∧ (∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S)

/-- Number of ideal subsets given relatively prime positive integers p and q -/
theorem number_of_ideal_subsets (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) :
  ∃ k : ℕ, (k = (1 / (p + q) : ℚ) * Nat.choose (p + q) p) :=
sorry

end number_of_ideal_subsets_l372_372426


namespace min_value_of_M_l372_372131

noncomputable def a (n : ℕ) : ℝ := 3 + 2 * (n - 1)
noncomputable def b (n : ℕ) : ℝ := 2 ^ (n - 1)
noncomputable def a_over_b (n : ℕ) : ℝ := a n / b n

noncomputable def T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a_over_b (i + 1))

theorem min_value_of_M (M : ℝ) (hM : ∀ n : ℕ, n > 0 → T n < M) :
  10 ≤ M :=
sorry

end min_value_of_M_l372_372131


namespace sqrt_sum_S_eq_624_div_25_l372_372434

theorem sqrt_sum_S_eq_624_div_25 :
  (∑ n in Finset.range 24, Real.sqrt (1 + 1 / n.succ ^ 2 + 1 / (n.succ + 1) ^ 2)) = 624 / 25 :=
by
  sorry

end sqrt_sum_S_eq_624_div_25_l372_372434


namespace rational_set_of_expression_l372_372515

def is_rational (x : ℚ) := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem rational_set_of_expression :
  ∀ (x : ℝ), is_rational (x + sqrt (x^2 + 1) - 1 / (x + sqrt (x^2 + 1))) ↔ is_rational x := 
  sorry

end rational_set_of_expression_l372_372515


namespace james_money_left_l372_372801

theorem james_money_left 
  (earnings_per_week : ℕ)
  (weeks_saved : ℕ)
  (video_game_fraction : ℝ)
  (book_fraction : ℝ)
  (h1 : earnings_per_week = 10)
  (h2 : weeks_saved = 4)
  (h3 : video_game_fraction = 1/2)
  (h4 : book_fraction = 1/4) : 
  let total_savings := earnings_per_week * weeks_saved in
  let after_video_game := total_savings * video_game_fraction in
  let after_book := after_video_game * book_fraction in
  let remaining_money := after_video_game - after_book in
  remaining_money = 15 :=
by
  sorry

end james_money_left_l372_372801


namespace solve_for_a_l372_372003

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372003


namespace quadratic_eq_D_l372_372169

-- Definitions of the equations as conditions
def eq_A (x y : ℝ) : Prop := x^2 + x - y = 0
def eq_B (a x : ℝ) : Prop := a * x^2 + 2 * x - 3 = 0
def eq_C (x : ℝ) : Prop := x^2 + 2 * x + 5 = x^3
def eq_D (x : ℝ) : Prop := x^2 - 1 = 0

-- Proving that eq_D is a quadratic equation in one variable
theorem quadratic_eq_D : ∀ x : ℝ, eq_D x → ∃ a b c : ℝ, a ≠ 0 ∧  eq_D x = (a * x^2 + b * x + c = 0) :=
by
  intros x hD
  use [1, 0, -1]
  constructor
  { -- Prove a ≠ 0
    norm_num, },
  { -- Prove the equation is of the form a*x^2 + b*x + c = 0
    exact hD }

end quadratic_eq_D_l372_372169


namespace range_of_m_l372_372341

def f (m x : ℝ) : ℝ :=
  if x < 2 then 2^(x - m)
  else (m * x) / (4 * x^2 + 16)

theorem range_of_m (m : ℝ) :
  (∀ (x1 : ℝ), x1 ≥ 2 → ∃ (x2 : ℝ), x2 < 2 ∧ f m x1 = f m x2) → m ≤ 4 :=
sorry

end range_of_m_l372_372341


namespace combined_resistance_parallel_l372_372186

theorem combined_resistance_parallel (R1 R2 R3 R : ℝ)
  (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6)
  (h4 : 1/R = 1/R1 + 1/R2 + 1/R3) :
  R = 15/13 := 
by
  sorry

end combined_resistance_parallel_l372_372186


namespace tangent_line_monotonic_intervals_l372_372340

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - (a + 1/a) * x + real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x - (a + 1/a) + 1/x

theorem tangent_line (x : ℝ) (y : ℝ) (a : ℝ) (h_a : a = 2) (h_f : y = f 1 2) : 
  x + 2*y + 3 = 0 := 
sorry

theorem monotonic_intervals (a : ℝ) (h_a_pos : a > 0) (h_a_neq_1 : a ≠ 1) :
  (∀ x : ℝ, (0 < x ∧ x < 1/a) ∨ (x > a) → f' x a > 0) ∧ 
  (∀ x : ℝ, (1/a < x ∧ x < a) → f' x a < 0) ∨
  (0 < a ∧ a < 1) :=
sorry

end tangent_line_monotonic_intervals_l372_372340


namespace rectangle_area_304_l372_372837

variables (A B C D K : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space K]

-- Conditions: K is the midpoint of side AB, and the area of the shaded part is 76 cm^2.
variable h_midpoint : K = midpoint A B
variable h_shaded_area : measure (shaded_region) = 76

-- Question: The area of the entire rectangle is 304 cm^2.
theorem rectangle_area_304 :
  measure (rectangle A B C D) = 304 :=
sorry

end rectangle_area_304_l372_372837


namespace amplitude_of_sine_function_l372_372611

theorem amplitude_of_sine_function :
  ∀ (a b c d : ℝ),
  (∀ (x : ℝ), 1 ≤ a * sin(b * x + c) + d ∧ a * sin(b * x + c) + d ≤ 7) →
  a = 3 :=
by
  intros a b c d h
  sorry

end amplitude_of_sine_function_l372_372611


namespace intersection_of_sets_l372_372359

def M : Set ℤ := { x | Int.natAbs x = 1 }
def N : Set ℤ := { x | (1 / 2) < 2^x ∧ 2^x < 4 }

theorem intersection_of_sets :
  M ∩ N = {1} := by
  sorry

end intersection_of_sets_l372_372359


namespace factorial_div_l372_372163

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_div : (factorial 4) / (factorial (4 - 3)) = 24 := by
  sorry

end factorial_div_l372_372163


namespace find_a_l372_372348
-- Import the required library to bring all necessary functionalities

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - (3 / 2) * x^2

-- Define the maximum condition
def max_cond (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≤ 1 / 6  -- Maximum value should not exceed 1/6

-- Define the interval condition
def interval_cond (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) → f a x ≥ 1 / 8  -- Function value should be >= 1/8 in the interval

-- The main theorem stating that under given conditions, a = 1
theorem find_a (a : ℝ) (max_cond a) (interval_cond a) : a = 1 :=
  sorry -- The proof is omitted


end find_a_l372_372348


namespace problem_statement_l372_372299

theorem problem_statement :
  (b = (log 1001 2) + (log 1001 3) + (log 1001 4) + (log 1001 5) + (log 1001 6)) ∧
  (c = (log 1001 10) + (log 1001 11) + (log 1001 12) + (log 1001 13) + (log 1001 14) + (log 1001 15)) →
  b - c = -1 - log 1001 5 :=
by 
  sorry

end problem_statement_l372_372299


namespace number_of_pairs_sold_l372_372949

theorem number_of_pairs_sold (total_sales : ℝ) (average_price : ℝ) (h_total_sales : total_sales = 588) (h_average_price : average_price = 9.8) : total_sales / average_price = 60 := 
by {
    rw [h_total_sales, h_average_price],
    norm_num,
}

end number_of_pairs_sold_l372_372949


namespace probability_fourth_smallest_six_l372_372276

theorem probability_fourth_smallest_six :
  let S := finset.range 16 \ {0} in
  let total_ways := (finset.card (S)).choose 8 in
  let favorable_ways := (finset.card (finset.range 6 \ {0})).choose 3 * (finset.card (finset.range 16 \ S \ {0, 1, 2, 3, 4, 5, 6})).choose 5 in
  let probability := (favorable_ways : ℚ) / total_ways in
  probability = 4 / 21 :=
by
  let S := finset.range 16 \ {0}
  let total_ways := (finset.card (S)).choose 8
  let favorable_ways := (finset.card (finset.range 6 \ {0})).choose 3 * (finset.card (finset.range 16 \ S \ {0, 1, 2, 3, 4, 5, 6})).choose 5
  let probability := (favorable_ways : ℚ) / total_ways
  show probability = 4 / 21
  sorry

end probability_fourth_smallest_six_l372_372276


namespace quadratic_inequality_solution_l372_372481

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5 * x + 6 < 0 ↔ 2 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l372_372481


namespace minimum_manhattan_distance_l372_372395

open Real

def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 2 + P.2^2 = 1

def line (Q : ℝ × ℝ) : Prop := 3 * Q.1 + 4 * Q.2 = 12

def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem minimum_manhattan_distance :
  ∃ P Q, ellipse P ∧ line Q ∧
    ∀ P' Q', ellipse P' → line Q' → manhattan_distance P Q ≤ manhattan_distance P' Q' :=
  sorry

end minimum_manhattan_distance_l372_372395


namespace richard_older_than_david_l372_372384

variable {R D S : ℕ}

theorem richard_older_than_david (h1 : R > D) (h2 : D = S + 8) (h3 : R + 8 = 2 * (S + 8)) (h4 : D = 14) : R - D = 6 := by
  sorry

end richard_older_than_david_l372_372384


namespace common_ratio_of_geometric_sequence_l372_372126

variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b_n (n + 1) = b_n n * r

def arithmetic_to_geometric (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  b_n 0 = a_n 2 ∧ b_n 1 = a_n 3 ∧ b_n 2 = a_n 7

-- Mathematical Proof Problem
theorem common_ratio_of_geometric_sequence :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), d ≠ 0 →
  is_arithmetic_sequence a_n d →
  (∃ (b_n : ℕ → ℝ) (r : ℝ), arithmetic_to_geometric a_n b_n ∧ is_geometric_sequence b_n r) →
  ∃ r, r = 4 :=
sorry

end common_ratio_of_geometric_sequence_l372_372126


namespace ellipse_eccentricity_square_l372_372333

theorem ellipse_eccentricity_square (b c a e : ℝ) 
  (h1 : b = c) 
  (h2 : a = sqrt (2 * b^2)) 
  (h3 : e = c / a) : 
  e = sqrt(2) / 2 :=
by
  sorry

end ellipse_eccentricity_square_l372_372333


namespace line_through_point_and_intersects_circle_with_chord_length_8_l372_372290

theorem line_through_point_and_intersects_circle_with_chord_length_8 :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = 0 ↔ x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) ↔ 
  (∃ (x : ℝ), x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) := 
by
  sorry

end line_through_point_and_intersects_circle_with_chord_length_8_l372_372290


namespace find_a_l372_372066

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372066


namespace tetrahedron_vertex_edges_form_triangle_l372_372467
-- Import the necessary library to bring in the entirety of Mathlib

-- Define the problem using conditions and the final proof goal
theorem tetrahedron_vertex_edges_form_triangle 
    (vertices : Set PointCardinal)
    (tetrahedron : TetrahedronStructure vertices)
    (edges : ∀ {v : vertices}, {u : vertices}, v ≠ u → ℝ)
    : ∃ (v : vertices), let es := {e : ℝ | ∃ (u : vertices), u ≠ v ∧ e = edges u v} 
    in set.toFinset es.card = 3 ∧ ((∑ e ∈ es) > 2 * (max e ∈ es)) := 
sorry

end tetrahedron_vertex_edges_form_triangle_l372_372467


namespace bacteria_count_correct_l372_372110

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 800

-- Define the doubling time in hours
def doubling_time : ℕ := 3

-- Define the function that calculates the number of bacteria after t hours
noncomputable def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * 2 ^ (t / doubling_time)

-- Define the target number of bacteria
def target_bacteria : ℕ := 51200

-- Define the specific time we want to prove the bacteria count equals the target
def specific_time : ℕ := 18

-- Prove that after 18 hours, there will be exactly 51,200 bacteria
theorem bacteria_count_correct : bacteria_after specific_time = target_bacteria :=
  sorry

end bacteria_count_correct_l372_372110


namespace intersection_A_B_l372_372770

def A : Set ℝ := { x | |x| > 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l372_372770


namespace find_a_l372_372063

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372063


namespace area_enclosed_by_curves_l372_372490

open Real

theorem area_enclosed_by_curves :
  ∫ x in 1..2, (x^2 - (1 / x)) = (7 / 3) - log 2 :=
by
  sorry

end area_enclosed_by_curves_l372_372490


namespace exist_interval_l372_372485

noncomputable def f (x : ℝ) := Real.log x + x - 4

theorem exist_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end exist_interval_l372_372485


namespace find_x_l372_372182

theorem find_x (x : ℝ) (h : 5.76 = 0.12 * 0.40 * x) : x = 120 := 
sorry

end find_x_l372_372182


namespace abs_z2_minus_2z_eq_2_l372_372750

theorem abs_z2_minus_2z_eq_2 (z : ℂ) (h : z = 1 + complex.I) : abs (z^2 - 2*z) = 2 := by
  sorry

end abs_z2_minus_2z_eq_2_l372_372750


namespace dhoni_remaining_earnings_l372_372273

theorem dhoni_remaining_earnings :
  let rent := 0.20
  let dishwasher := 0.15
  let bills := 0.10
  let car := 0.08
  let grocery := 0.12
  let tax := 0.05
  let expenses := rent + dishwasher + bills + car + grocery + tax
  let remaining_after_expenses := 1.0 - expenses
  let savings := 0.40 * remaining_after_expenses
  let remaining_after_savings := remaining_after_expenses - savings
  remaining_after_savings = 0.18 := by
sorry

end dhoni_remaining_earnings_l372_372273


namespace symmetry_center_of_f_l372_372342

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x + π / 3) + sqrt 3 * sin (ω * x - π / 6)

theorem symmetry_center_of_f :
  ∀ (ω : ℝ) (hω : ω > 0), 
  (∀ x : ℝ, (f ω (x + π) = f ω x)) →
  (f ω (π / 2) = 0) →
  (∀ x : ℝ, f ω (x + π / 2) = f ω (π / 2 - x)) :=
by 
  intros ω hω h_period h_f_pi2

  -- Using the periodicity of the function and the given conditions
  sorry

end symmetry_center_of_f_l372_372342


namespace complex_magnitude_example_l372_372739

theorem complex_magnitude_example (z : ℂ) (h : z = 1 + Complex.i) : Complex.abs (z^2 - 2*z) = 2 := 
by 
  rw [h]
  sorry

end complex_magnitude_example_l372_372739


namespace female_population_correct_l372_372385

def total_population : ℕ := 728400
def migrants_percentage : ℝ := 0.35
def rural_migrants_percentage : ℝ := 0.20
def local_females_percentage : ℝ := 0.48
def rural_migrants_females_percentage : ℝ := 0.30
def urban_migrants_females_percentage : ℝ := 0.40

def migrants_population : ℝ := migrants_percentage * total_population
def local_population : ℝ := total_population - migrants_population
def rural_migrants_population : ℝ := rural_migrants_percentage * migrants_population
def urban_migrants_population : ℝ := migrants_population - rural_migrants_population

def local_females_population : ℝ := local_females_percentage * local_population
def rural_migrants_females_population : ℝ := rural_migrants_females_percentage * rural_migrants_population
def urban_migrants_females_population : ℝ := urban_migrants_females_percentage * urban_migrants_population

def total_female_population := local_females_population + rural_migrants_females_population + urban_migrants_females_population

theorem female_population_correct : total_female_population = 324128 := 
by {
  calculate sorry 
}

end female_population_correct_l372_372385


namespace find_a_l372_372768

def op_e : ℕ → ℕ → ℕ := λ x y, 2 * x * y

theorem find_a (a : ℕ) : (op_e 4 5 = 40) → (op_e a 40 = 640) → a = 8 :=
by
  intros h1 h2
  sorry

end find_a_l372_372768


namespace question_I_question_II_l372_372344

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem question_I (h_extreme : ∃ x, x = 2 ∧ IsExtreme (f (3 / 2) x)) : ∃ a, a = 3 / 2 :=
  sorry

theorem question_II (h_nonneg : ∀ x, x ∈ Icc (1:ℝ) (Real.top) → f a x ≥ 0) : a ≤ 1 :=
  sorry

end question_I_question_II_l372_372344


namespace number_of_valid_pairs_l372_372221

theorem number_of_valid_pairs (m n : ℕ) (h1 : n > m) (h2 : 3 * (m - 4) * (n - 4) = m * n) : 
  (m, n) = (7, 18) ∨ (m, n) = (8, 12) ∨ (m, n) = (9, 10) ∨ (m-6) * (n-6) = 12 := sorry

end number_of_valid_pairs_l372_372221


namespace absolute_value_z_squared_minus_2z_l372_372758

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem
theorem absolute_value_z_squared_minus_2z : complex.abs (z^2 - 2*z) = 2 := by
  sorry

end absolute_value_z_squared_minus_2z_l372_372758


namespace cyclic_quadrilateral_perpendicular_diagonals_l372_372867

/-- 
  Proving the relationship between a quadrilateral formed by circle centers and another 
  quadrilateral formed by points of tangency of these circles.
-/
theorem cyclic_quadrilateral_perpendicular_diagonals 
  (O_1 O_2 O_3 O_4 P_1 P_2 P_3 P_4 : Type)
  (circle1 : circle O_1)
  (circle2 : circle O_2)
  (circle3 : circle O_3)
  (circle4 : circle O_4)
  (tangent_points : circle1.tangent P_1 → circle2.tangent P_2 → circle3.tangent P_3 → circle4.tangent P_4)
  (external_tangents : circle1.externally_touches circle2 → circle2.externally_touches circle3 → circle3.externally_touches circle4 → circle4.externally_touches circle1) :
  ¬ (is_cyclic_quadrilateral O_1 O_2 O_3 O_4 ↔ are_perpendicular (diagonal P_1 P_3) (diagonal P_2 P_4)) :=
sorry

end cyclic_quadrilateral_perpendicular_diagonals_l372_372867


namespace find_a_l372_372056

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372056


namespace ice_cream_nieces_l372_372534

theorem ice_cream_nieces :  ∃ n : ℕ, 143 = 13 * n ∧ n = 11 :=
by
  use 11
  sorry

end ice_cream_nieces_l372_372534


namespace product_of_t_values_l372_372660

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372660


namespace problem1_problem2_problem3_l372_372373

-- Definitions and conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Question 1: Prove the function is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

-- Question 2: Prove the function is monotonically decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

-- Question 3: Solve the inequality given f(2) = 1
theorem problem3 (h3 : f 2 = 1) : ∀ x : ℝ, f (-x^2) + 2*f x + 4 < 0 ↔ -2 < x ∧ x < 4 := by
  sorry

end problem1_problem2_problem3_l372_372373


namespace quadrilateral_ABCD_l372_372990

theorem quadrilateral_ABCD (AB BC CD BE CE BD x : ℝ)
  (h1 : ∠B = 90) (h2 : ∠C = 90) 
  (h3 : ΔABC ~ ΔBCD) (h4 : ΔABC ~ ΔCEB)
  (h5 : area (ΔAED) = 9 * area (ΔCEB))
  (h6 : AB > BC)
  : AB / BC = Real.sqrt (4 + 2 * Real.sqrt 3) :=
sorry

end quadrilateral_ABCD_l372_372990


namespace crayons_taken_out_l372_372142

-- Define the initial and remaining number of crayons
def initial_crayons : ℕ := 7
def remaining_crayons : ℕ := 4

-- Define the proposition to prove
theorem crayons_taken_out : initial_crayons - remaining_crayons = 3 := by
  sorry

end crayons_taken_out_l372_372142


namespace solve_system1_solve_system2_l372_372852

-- Definitions for the first system of equations
def system1_equation1 (x y : ℚ) := 3 * x - 6 * y = 4
def system1_equation2 (x y : ℚ) := x + 5 * y = 6

-- Definitions for the second system of equations
def system2_equation1 (x y : ℚ) := x / 4 + y / 3 = 3
def system2_equation2 (x y : ℚ) := 3 * (x - 4) - 2 * (y - 1) = -1

-- Lean statement for proving the solution to the first system
theorem solve_system1 :
  ∃ (x y : ℚ), system1_equation1 x y ∧ system1_equation2 x y ∧ x = 8 / 3 ∧ y = 2 / 3 :=
by
  sorry

-- Lean statement for proving the solution to the second system
theorem solve_system2 :
  ∃ (x y : ℚ), system2_equation1 x y ∧ system2_equation2 x y ∧ x = 6 ∧ y = 9 / 2 :=
by
  sorry

end solve_system1_solve_system2_l372_372852


namespace ellipse_a_value_l372_372033

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372033


namespace pure_gala_trees_l372_372210

variables (T F G : ℕ)

theorem pure_gala_trees :
  (0.1 * T : ℝ) + F = 238 ∧ F = (3 / 4) * ↑T → G = T - F → G = 70 :=
by
  intro h
  sorry

end pure_gala_trees_l372_372210


namespace find_certain_number_l372_372081

noncomputable def certain_number (x : ℝ) : Prop :=
  (0.4 * x - (1 / 3) * (0.4 * x) = 48)

theorem find_certain_number (x : ℝ) (h : certain_number x) : x ≈ 179.94 :=
  sorry

end find_certain_number_l372_372081


namespace find_prime_triples_l372_372643

theorem find_prime_triples :
  ∀ p x y : ℤ, p.prime ∧ x > 0 ∧ y > 0 ∧ p^x = y^3 + 1 →
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) := by
  sorry

end find_prime_triples_l372_372643


namespace compare_exponential_functions_l372_372256

theorem compare_exponential_functions (x : ℝ) (hx1 : 0 < x) :
  0.4^4 < 1 ∧ 1 < 4^0.4 :=
by sorry

end compare_exponential_functions_l372_372256


namespace intersection_point_value_l372_372878

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 10

theorem intersection_point_value : ∃ c : ℝ, (f 7 = c) ∧ (f c = 7) ∧ c = 87 :=
by
  use 87
  split
  {
    -- c = f(7)
    calc f 7 = 2 * 7^2 - 3 * 7 + 10 : by simp [f]
       ... = 98 - 21 + 10          : by norm_num
       ... = 87                    : by norm_num
  }
  split
  {
    -- f(c) = 7
    calc f 87 = 2 * 87^2 - 3 * 87 + 10 : by simp [f]
         ... = 2 * 7569 - 261 + 10     : by norm_num
         ... = 15138 - 261 + 10        : by norm_num
         ... = 14887                   : by norm_num
  }
  {
    -- c = 87 already established
    exact rfl
  }

end intersection_point_value_l372_372878


namespace periodic_function_l372_372838

variable {f : ℝ → ℝ}

theorem periodic_function (h : ∀ x : ℝ, f(x + 1) + f(x - 1) = sqrt 3 * f x) (h_nonconst : ∃ x y : ℝ, x ≠ y ∧ f x ≠ f y) : 
    ∃ p > 0, (∀ x : ℝ, f (x + p) = f x) ∧ p = 12 :=
sorry

end periodic_function_l372_372838


namespace total_wicks_l372_372959

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l372_372959


namespace find_base_b_l372_372780

def base_b_square (b : ℤ) : ℤ :=
  (3 * b + 5) ^ 2

def base_b_expansion (b : ℤ) : ℤ :=
  b ^ 3 + 3 * b ^ 2 + 3 * b + 1

theorem find_base_b :
  ∃ b : ℤ, base_b_square b = base_b_expansion b ∧ b = 12 :=
by {
  use 12,
  rw [base_b_square, base_b_expansion], -- expand the definitions
  -- use specific tactics and arithmetic to finish the proof
  sorry
}

end find_base_b_l372_372780


namespace total_boys_slide_l372_372148

theorem total_boys_slide (initial_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 :=
by
  sorry

end total_boys_slide_l372_372148


namespace right_triangle_area_l372_372390

theorem right_triangle_area (A B C D : Point) (BD DC : ℝ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : BD = 4) (h5 : DC = 5)
  (h6 : RightTriangle A B C)
  (h7 : AngleBisector A (A ↔︎ B) D ∧ AngleBisector A (A ↔︎ C) D ∧ B ↔︎ C = BD + DC) :
  Area A B C = 54 :=
by
  sorry

end right_triangle_area_l372_372390


namespace ellipse_eccentricity_a_l372_372013

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372013


namespace tetrahedron_ratio_sums_to_one_l372_372942

noncomputable def tetra_intersection_ratios (A B C D P A1 B1 C1 D1 : Point) : Prop :=
  ∃ (in_tetrahedron : point_in_tetrahedron P A B C D)
    (intersects_A1B1 : line_intersects_face P A (face B C D) A1)
    (intersects_B1C1 : line_intersects_face P B (face A C D) B1)
    (intersects_C1A1 : line_intersects_face P C (face A B D) C1)
    (intersects_D1A1 : line_intersects_face P D (face A B C) D1),
  (P.distance_to A1 / A.distance_to A1 +
   P.distance_to B1 / B.distance_to B1 +
   P.distance_to C1 / C.distance_to C1 +
   P.distance_to D1 / D.distance_to D1) = 1

-- Definitions for geometric objects and relationships used in the def above. 
def Point := ℝ × ℝ × ℝ     -- Assume 3D cartesian space for points
def Face := Point × Point × Point    -- A face is a triangle
def Tetrahedron := Point × Point × Point × Point     -- A tetrahedron is defined by its 4 vertices

def point_in_tetrahedron (P A B C D : Point) : Prop := sorry
def line_intersects_face (P Q : Point) (f : Face) (R : Point) : Prop := sorry
def face (A B C : Point) : Face := (A, B, C)
def Point.distance_to (P Q : Point) : ℝ := sorry     -- Euclidean distance between P and Q
def Tetrahedron.volume (A B C D : Point) : ℝ := sorry

-- Main theorem statement of the problem.
theorem tetrahedron_ratio_sums_to_one (A B C D P A1 B1 C1 D1 : Point) :
  tetra_intersection_ratios A B C D P A1 B1 C1 D1 :=
sorry

end tetrahedron_ratio_sums_to_one_l372_372942


namespace rice_yield_prediction_l372_372378

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 5 * x + 250

-- Define the specific condition for x = 80
def fertilizer_amount : ℝ := 80

-- State the theorem for the expected rice yield
theorem rice_yield_prediction : regression_line fertilizer_amount = 650 :=
by
  sorry

end rice_yield_prediction_l372_372378


namespace proof_largest_possible_value_e_l372_372815

noncomputable def largest_possible_value_e (PQ X Y Z : Point) (diameter : ℝ) (midpoint_X : is_midpoint X PQ) 
  (length_PY : ℝ) (arc_condition : OnSemicircle X Y PQ diameter X) : ℝ :=
  let S := intersection PQ XZ in
  let T := intersection PQ YZ in
  let e := segment_length S T in
  let β := angle PZQ in
  let cos_value := 3/real.sqrt 13 in
  if β = real.acos cos_value then 
    (13 - 6 * real.sqrt 5)
  else 
    0 -- This default is arbitrary and must be justified further in proofs

-- The statement you need to prove
theorem proof_largest_possible_value_e :
  ∀ (PQ X Y Z : Point) (diameter : ℝ) (midpoint_X : is_midpoint X PQ) 
  (length_PY : ℝ) (arc_condition : OnSemicircle X Y PQ diameter X), 
  length_PY = 4/5 → 
  e = 13 - 6 * real.sqrt 5 :=
begin
  sorry -- Proof will go here
end

end proof_largest_possible_value_e_l372_372815


namespace positive_solution_sqrt_a_sub_b_l372_372625

theorem positive_solution_sqrt_a_sub_b (a b : ℕ) (x : ℝ) 
  (h_eq : x^2 + 14 * x = 32) 
  (h_form : x = Real.sqrt a - b) 
  (h_pos_nat : a > 0 ∧ b > 0) : 
  a + b = 88 := 
by
  sorry

end positive_solution_sqrt_a_sub_b_l372_372625


namespace local_value_of_3_in_2345_l372_372541

theorem local_value_of_3_in_2345 :
  let local_value_2 := 2000
  let local_value_4 := 40
  let local_value_5 := 5
  let total_sum := 2345
  let known_sum := local_value_2 + local_value_4 + local_value_5
  in
  total_sum - known_sum = 300 := by
  sorry

end local_value_of_3_in_2345_l372_372541


namespace ticket_cost_proof_l372_372608

def adult_ticket_price : ℕ := 55
def child_ticket_price : ℕ := 28
def senior_ticket_price : ℕ := 42

def num_adult_tickets : ℕ := 4
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℕ :=
  (num_adult_tickets * adult_ticket_price) + (num_child_tickets * child_ticket_price) + (num_senior_tickets * senior_ticket_price)

theorem ticket_cost_proof : total_ticket_cost = 318 := by
  sorry

end ticket_cost_proof_l372_372608


namespace six_digit_number_count_l372_372882

theorem six_digit_number_count :
  ∃ (count : ℕ), count = 60 ∧ 
  ∀ (digits : list ℕ), digits = [0, 1, 2, 3, 4, 5] →
    (∀ (x : ℕ), x ∈ digits → x.bodd = tt ∨ x.bodd = ff ) →
    (∀ (x y : ℕ), x ≠ y → x ∈ digits → y ∈ digits → x ≠ y) →
  ∀ (num : list ℕ), num.length = 6 →
    (∀ (n : ℕ), n < 6 → (num.nth n).isSome) →
    (∀ (n : ℕ), n < 5 → (num.nth n).iget.bodd ≠ (num.nth (n+1)).iget.bodd) →
    list.perm_eq digits (num.map (λ n, n)) →
    count = nat.factorial 3 ^ 2 + 2 * 2 * nat.factorial 3 :=
by
  sorry

end six_digit_number_count_l372_372882


namespace comparison_of_quantities_l372_372688

theorem comparison_of_quantities
  (α β γ : ℝ)
  (hα : 0 < α ∧ α < 1) 
  (hβ : 0 < β ∧ β < 1) 
  (hγ : 0 < γ ∧ γ < 1) :
  let f (a b c : ℝ) := (a + b + c) / 3 in
  let x := f α β γ in
  let y := f (Real.arccos α) (Real.arccos β) (Real.arccos γ) in
  let z := Real.arccos (f α β γ) in
  let w := Real.cos (f α β γ) in
  let t := f (Real.cos α) (Real.cos β) (Real.cos γ) in
  y > z ∧ z > x ∧ x > w ∧ w > t :=
by
  sorry

end comparison_of_quantities_l372_372688


namespace Clarence_total_oranges_l372_372254

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l372_372254


namespace factorization_correct_l372_372552

theorem factorization_correct :
    (∀ (x y : ℝ), x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y)) :=
by
  intro x y
  sorry

end factorization_correct_l372_372552


namespace find_a_l372_372067

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372067


namespace part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l372_372695

noncomputable def A : Set ℝ := { x : ℝ | 3 < x ∧ x < 10 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 9 * x + 14 < 0 }
noncomputable def C (m : ℝ) : Set ℝ := { x : ℝ | 5 - m < x ∧ x < 2 * m }

theorem part_I_A_inter_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } :=
sorry

theorem part_I_complement_A_union_B :
  (Aᶜ) ∪ B = { x : ℝ | x < 7 ∨ x ≥ 10 } :=
sorry

theorem part_II_range_of_m :
  {m : ℝ | C m ⊆ A ∩ B} = {m : ℝ | m ≤ 2} :=
sorry

end part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l372_372695


namespace volume_fraction_l372_372224

variable (base_edge : ℝ) (original_altitude : ℝ) (smaller_altitude : ℝ)
variable (original_volume : ℝ) (smaller_volume : ℝ) (remaining_fraction : ℝ)

-- Defines the conditions as constants/variables
def base_edge := 40
def original_altitude := 18
def smaller_altitude := original_altitude / 3

-- Defines the volumes based on the given properties
def original_volume := (1 / 3) * (base_edge ^ 2) * original_altitude
def smaller_base_edge := base_edge / 3
def smaller_volume := (1 / 3) * (smaller_base_edge ^ 2) * smaller_altitude

-- Statement for the remaining volume fraction of the frustum
theorem volume_fraction (h1 : original_volume = (1 / 3) * (40 ^ 2) * 18)
                        (h2 : smaller_volume = (1 / 3) * ((40 / 3) ^ 2) * (18 / 3)) :
  remaining_fraction = 1 - (smaller_volume / original_volume) := 
begin
  sorry
end

end volume_fraction_l372_372224


namespace probability_sum_multiple_of_3_l372_372898

def balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def possible_pairs (x : ℕ) (y : ℕ) : Prop :=
  x ≠ y ∧ x ∈ balls ∧ y ∈ balls

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def favorable_pairs : set (ℕ × ℕ) :=
  { (x, y) ∈ (balls.product balls) | possible_pairs x y ∧ is_multiple_of_3 (x + y) }

noncomputable def total_pairs : set (ℕ × ℕ) :=
  { (x, y) ∈ (balls.product balls) | x ≠ y }

theorem probability_sum_multiple_of_3 (total_outcomes favorable_outcomes : ℚ) :
  total_outcomes = 132 ∧ favorable_outcomes = 34 →
  (favorable_outcomes / total_outcomes = 17 / 66) :=
by
    sorry

end probability_sum_multiple_of_3_l372_372898


namespace sara_nada_house_size_l372_372092

theorem sara_nada_house_size (m : ℝ) :
  let sara_house := 1000
  let nada_house := 450
  let extra_size := 100
  sara_house = nada_house * m + extra_size -> m = 2 :=
begin
  sorry
end

end sara_nada_house_size_l372_372092


namespace abs_z_squared_minus_two_z_eq_two_l372_372748

theorem abs_z_squared_minus_two_z_eq_two (z : ℂ) (hz : z = 1 + 1*Complex.i) : 
  |z^2 - 2*z| = 2 :=
begin
  sorry
end

end abs_z_squared_minus_two_z_eq_two_l372_372748


namespace product_of_roots_eq_negative_forty_nine_l372_372655

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372655


namespace factorize_polynomial_l372_372282

theorem factorize_polynomial (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := 
by
  sorry

end factorize_polynomial_l372_372282


namespace sample_size_l372_372896

variable (num_classes : ℕ) (papers_per_class : ℕ)

theorem sample_size (h_classes : num_classes = 8) (h_papers : papers_per_class = 12) : 
  num_classes * papers_per_class = 96 := 
by 
  sorry

end sample_size_l372_372896


namespace min_value_inequality_l372_372441

variable (x y : ℝ)

theorem min_value_inequality (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ m : ℝ, m = 1 / 4 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (x ^ 2) / (x + 2) + (y ^ 2) / (y + 1) ≥ m) :=
by
  use (1 / 4)
  sorry

end min_value_inequality_l372_372441


namespace find_a_l372_372017

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372017


namespace probability_distance_at_least_one_l372_372435

theorem probability_distance_at_least_one (side_length : ℝ) (length_condition : side_length = 2) :
  let P := ((15 - Real.pi) / 8)
  in P = (probability_involving_random_points side_length) := 
sorry

end probability_distance_at_least_one_l372_372435


namespace percentage_within_one_sd_l372_372388

variable (a d : ℝ)
variable (distribution : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- Conditions
def symmetric_about_mean (a : ℝ) := 
  ∀ x, distribution (a + x) = distribution (a - x)

def percentage_less_than (p b : ℝ) :=
  P (∫ x in -∞..b, distribution x) = p

-- Proof statement
theorem percentage_within_one_sd (h1 : symmetric_about_mean a)
                                 (h2 : percentage_less_than 0.84 (a + d)) :
  percentage_within (a - d) (a + d) 0.68 :=
  sorry

end percentage_within_one_sd_l372_372388


namespace complex_expression_value_l372_372196

theorem complex_expression_value : ( ( ( Complex.sqrt 2 ) / (1 - Complex.I) ) ^ 2018 + ( (1 + Complex.I) / (1 - Complex.I) ) ^ 6 ) = -1 + Complex.I := sorry

end complex_expression_value_l372_372196


namespace binomial_sum_real_part_l372_372984

theorem binomial_sum_real_part :
  (1 / 2 ^ 1988) * (∑ n in Finset.range 995, (-3) ^ n * Nat.choose 1988 (2 * n)) = -1 / 2 :=
by
  sorry

end binomial_sum_real_part_l372_372984


namespace surface_area_of_sphere_l372_372519

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ A : ℝ, A = 36 * π * (2^(2/3)) := by 
  sorry

end surface_area_of_sphere_l372_372519


namespace james_savings_l372_372798

theorem james_savings (weekly_allowance : ℕ) (weeks : ℕ) (video_game_fraction : ℕ) (book_fraction : ℕ) :
  weekly_allowance = 10 →
  weeks = 4 →
  video_game_fraction = 2 →
  book_fraction = 4 →
  let total_savings := weekly_allowance * weeks in
  let after_video_game := total_savings / video_game_fraction in
  let remaining_after_video_game := total_savings - after_video_game in
  let after_book := remaining_after_video_game / book_fraction in
  let final_amount := remaining_after_video_game - after_book in
  final_amount = 15 :=
by
  intros hw ha hv hb
  rw [hw, ha, hv, hb]
  let total_savings := weekly_allowance * weeks 
  let after_video_game := total_savings / video_game_fraction
  let remaining_after_video_game := total_savings - after_video_game
  let after_book := remaining_after_video_game / book_fraction
  let final_amount := remaining_after_video_game - after_book
  have h1 : total_savings = 40 := by
    rw [hw, ha]
  have h2 : after_video_game = 20 := by
    rw [hv, h1]
  have h3 : remaining_after_video_game = 20 := by
    exact sub_eq_of_eq_add' (by rw [hv, h1])
  have h4 : after_book = 5 := by
    rw [hb, h3]
  have h5 : final_amount = 15 := by
    exact sub_eq_of_eq_add (by rw [hb, h3])
  exact h5

end james_savings_l372_372798


namespace a4_eq_2_or_neg2_l372_372298

variable (a : ℕ → ℝ)
variable (r : ℝ)

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
axiom h1 : is_geometric_sequence a r
axiom h2 : a 2 * a 6 = 4

-- Theorem to prove
theorem a4_eq_2_or_neg2 : a 4 = 2 ∨ a 4 = -2 :=
sorry

end a4_eq_2_or_neg2_l372_372298


namespace height_of_first_tank_l372_372150

-- Define variables representing the emptying times and heights of the tanks.
variable (t1 t2 : ℝ) -- t1 is time in hours for tank 1, t2 is time in hours for tank 2
variable (h1 h2 : ℝ) -- h1 is the height of tank 1, h2 is the height of tank 2

-- Define the conditions from the problem.
-- t1 = 5 hours (17:00 - 12:00)
-- t2 = 8 hours (20:00 - 12:00)
-- h2 = 10 meters

axiom t1_eq_5 : t1 = 5
axiom t2_eq_8 : t2 = 8
axiom h2_eq_10 : h2 = 10

-- Define the intermediate relationships
-- At 2 PM (14:00), both tanks have the same water level.
-- For tank 1: (1 - 2 / t1) * h1 = (1 - 2 / t2) * h2

theorem height_of_first_tank : h1 = 12.5 :=
by
  have h1_eq : (1 - 2 / t1) * h1 = (1 - 2 / t2) * h2 := by sorry
  simp [t1_eq_5, t2_eq_8] at h1_eq -- substitute t1 and t2 values
  sorry

end height_of_first_tank_l372_372150


namespace num_of_chairs_per_row_l372_372680

theorem num_of_chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (chairs_per_row : ℕ)
  (h1 : total_chairs = 432)
  (h2 : num_rows = 27) :
  total_chairs = num_rows * chairs_per_row ↔ chairs_per_row = 16 :=
by
  sorry

end num_of_chairs_per_row_l372_372680


namespace sequence_relation_l372_372513

def a_sequence : ℕ → ℕ
| 1       := 0
| 2       := 1
| n + 1   := (n * a_sequence n + n * (n - 1) * a_sequence (n - 1) + (-1) ^ (n - 1) * n) / 2 + (-1) ^ n

def sum_binomial_coefficients (n : ℕ) : ℕ :=
∑ k in Finset.range n, (k + 1) * Nat.choose n k * a_sequence (n - k)

theorem sequence_relation (n : ℕ) : 
  (n > 0) → (a_sequence n + sum_binomial_coefficients n = 2 * Nat.factorial n - (n + 1)) :=
by
  intros
  sorry

end sequence_relation_l372_372513


namespace product_of_t_values_l372_372664

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372664


namespace probability_prime_1_to_30_l372_372538

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem probability_prime_1_to_30 :
  let total_numbers := 30
  let prime_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 30 ∧ is_prime n}.card
  (prime_numbers : ℚ) / (total_numbers : ℚ) = 1 / 3 :=
by
  -- Definitions
  let total_numbers := 30
  let prime_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 30 ∧ is_prime n}.card
  -- Proof will be filled here
  sorry

end probability_prime_1_to_30_l372_372538


namespace sequence_sum_l372_372635

theorem sequence_sum (x y : ℕ) 
  (r : ℚ) 
  (h1 : 4 * r = 1) 
  (h2 : x = 256 * r)
  (h3 : y = x * r): 
  x + y = 80 := 
by 
  sorry

end sequence_sum_l372_372635


namespace symmetric_line_x_axis_l372_372266

theorem symmetric_line_x_axis (x y : ℝ) :
  let L := 2 * x - 3 * y + 2
  in L = 0 ↔ (2 * x + 3 * y + 2) = 0 :=
sorry

end symmetric_line_x_axis_l372_372266


namespace min_M_value_l372_372130

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}
variable {M : ℕ}

def arithmetic_seq (n : ℕ) := 2 * n + 1
def geometric_seq (n : ℕ) := 2 ^ (n - 1)

noncomputable def T (n : ℕ) := ∑ i in finset.range n, (arithmetic_seq i / geometric_seq i)

theorem min_M_value : (∀ n : ℕ, T n < 10) → M ≥ 10 :=
sorry

end min_M_value_l372_372130


namespace find_inclination_angle_of_line_l372_372112

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Define the equation of the line
def line_eq (k : ℝ) : ℝ := 3

-- Define the distance from the center to the line
def distance_from_center_to_line (k : ℝ) : ℝ := |2 * k| / Real.sqrt (k^2 + 1)

-- Define the chord length condition
def chord_length_condition (k : ℝ) : Prop := 2 * Real.sqrt (circle_radius^2 - (distance_from_center_to_line k)^2) = 2 * Real.sqrt (3)

-- Define the angles we need to prove
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (±(Real.sqrt (3) / 3))

-- Define the main proof problem statement
theorem find_inclination_angle_of_line (k : ℝ):
  chord_length_condition k → inclination_angle (Real.arctan k) :=
begin
  sorry
end

end find_inclination_angle_of_line_l372_372112


namespace complement_intersection_l372_372725

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

#check (Set.compl B) ∩ A = {1}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 5}) (hB : B = {2, 3, 5}) :
   (U \ B) ∩ A = {1} :=
by
  sorry

end complement_intersection_l372_372725


namespace product_of_values_t_squared_eq_49_l372_372648

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372648


namespace speed_of_first_train_correct_l372_372152

def speed_of_first_train
  (len1 len2 : ℝ) -- lengths of the trains in meters
  (speed2 : ℝ) -- speed of the second train in kmph
  (time_to_clear_each_other : ℝ) -- time to clear each other in seconds
  (distance : ℝ := len1 + len2) -- total distance the trains need to clear each other in meters
  (distance_km : ℝ := distance / 1000) -- total distance in kilometers
  (time_hours : ℝ := time_to_clear_each_other / 3600) -- time to clear each other in hours) 
  (relative_speed : ℝ := distance_km / time_hours) -- relative speed in kmph
  (V1 : ℝ := relative_speed - speed2) -- speed of the first train in kmph
  : Prop :=
  V1 = 42

theorem speed_of_first_train_correct :
  speed_of_first_train 100 220 30 15.99872010239181 :=
by {
  norm_num,
  sorry
}

end speed_of_first_train_correct_l372_372152


namespace find_a_l372_372065

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372065


namespace cyclist_wait_time_l372_372880

theorem cyclist_wait_time
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time_minutes : ℝ)
  (hiker_speed_cond : hiker_speed = 4)
  (cyclist_speed_cond : cyclist_speed = 10)
  (wait_time_cond : wait_time_minutes = 5) :
  let hiker_speed_mpm := hiker_speed / 60
  let cyclist_speed_mpm := cyclist_speed / 60
  let distance := cyclist_speed_mpm * wait_time_minutes
  let time_for_hiker := distance / hiker_speed_mpm
  time_for_hiker = 12.5 :=
by {
  -- Convert speeds to miles per minute
  have hiker_speed_mpm_eq := calc hiker_speed / 60 = 4 / 60 : by rw hiker_speed_cond; rfl,
  have cyclist_speed_mpm_eq := calc cyclist_speed / 60 = 10 / 60 : by rw cyclist_speed_cond; rfl,
  
  -- Calculate the distance covered by the cyclist in the given wait time
  have distance_eq := calc cyclist_speed_mpm * wait_time_minutes = (10 / 60) * 5 : by rw [cyclist_speed_mpm_eq, wait_time_cond]; rfl,

  -- Calculate the time it takes for the hiker to cover the distance
  have time_for_hiker_eq := calc distance / (hiker_speed / 60) = (5/6) / (1/15) : by rw [distance_eq, hiker_speed_mpm_eq]; rfl
                          ... = 75 / 6 : by norm_num
                          ... = 12.5 : by norm_num,

  exact time_for_hiker_eq
}

end cyclist_wait_time_l372_372880


namespace axis_of_symmetry_and_increasing_interval_sin_alpha_value_l372_372709

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin x ^ 2

theorem axis_of_symmetry_and_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, x = π / 3 + k * (π / 2) → (f x = f (π / 3 + k * (π / 2)))) ∧
  (∀ k : ℤ, ∀ x : ℝ, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π → f'(x) > 0) :=
begin
  sorry
end

theorem sin_alpha_value (α : ℝ) (hα₁ : 0 < α) (hα₂ : α < π / 2) (h : f (α / 2) = 3 / 4) : 
  sin α = (sqrt 15 - sqrt 3) / 8 :=
begin
  sorry
end

end axis_of_symmetry_and_increasing_interval_sin_alpha_value_l372_372709


namespace trigonometric_expression_equals_correct_answer_l372_372305

variable (α : ℝ)

-- Conditions given in the problem
def condition_1 : Prop := 
  (cos α + sin α) / (cos α - sin α) = 2

-- Statement to prove: Given the conditions, the expression equals the correct answer
theorem trigonometric_expression_equals_correct_answer (h : condition_1 α) : 
  1 + 3 * sin α * cos α - 2 * cos α ^ 2 = 1 / 10 := 
sorry

end trigonometric_expression_equals_correct_answer_l372_372305


namespace solve_quadratic_l372_372098

theorem solve_quadratic : 
  (∃ x₁ x₂ : ℝ, x₁ = -1 + sqrt 6 / 2 ∧ x₂ = -1 - sqrt 6 / 2 ∧ 
    ∀ x : ℝ, 2 * x^2 + 4 * x - 1 = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end solve_quadratic_l372_372098


namespace constant_term_in_expansion_is_neg_42_l372_372869

-- Define the general term formula for (x - 1/x)^8
def binomial_term (r : ℕ) : ℤ :=
  (Nat.choose 8 r) * (-1 : ℤ) ^ r

-- Define the constant term in the product expansion
def constant_term : ℤ := 
  binomial_term 4 - 2 * binomial_term 5 

-- Problem statement: Prove the constant term is -42
theorem constant_term_in_expansion_is_neg_42 :
  constant_term = -42 := 
sorry

end constant_term_in_expansion_is_neg_42_l372_372869


namespace length_of_segment_in_cube_l372_372682

noncomputable def length_segment_in_cube_4 (X Y : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2 + (Y.3 - X.3)^2)

theorem length_of_segment_in_cube :
  ∀ (X Y : ℝ × ℝ × ℝ),
    (X = (0, 0, 0)) →
    (Y = (5, 5, 14)) →
    length_segment_in_cube_4 (0, 0, 5) (4, 4, 9) = 4 * real.sqrt 3 :=
by
  sorry

end length_of_segment_in_cube_l372_372682


namespace left_handed_and_like_scifi_count_l372_372140

-- Definitions based on the problem conditions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def like_scifi_members : ℕ := 18
def right_handed_not_like_scifi : ℕ := 4

-- Main proof statement
theorem left_handed_and_like_scifi_count :
  ∃ x : ℕ, (left_handed_members - x) + (like_scifi_members - x) + x + right_handed_not_like_scifi = total_members ∧ x = 4 :=
by
  use 4
  sorry

end left_handed_and_like_scifi_count_l372_372140


namespace complement_in_U_l372_372724

def U : Set ℕ := { x | 1 < x ∧ x < 4 }
def A : Set ℝ := { x | x^2 + 4 = 4 * x }
def CU (U A : Set α) : Set α := { x | x ∈ U ∧ x ∉ A }

theorem complement_in_U : CU U A = {3} := by
  sorry

end complement_in_U_l372_372724


namespace find_range_of_function_l372_372926

variable (a : ℝ) (x : ℝ)

def func (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem find_range_of_function (a : ℝ) :
  if a < 0 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -1 ≤ y ∧ y ≤ 3 - 4*a
  else if 0 ≤ a ∧ a ≤ 1 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ 3 - 4*a
  else if 1 < a ∧ a ≤ 2 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ -1
  else
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ 3 - 4*a ≤ y ∧ y ≤ -1
:= sorry

end find_range_of_function_l372_372926


namespace smallest_perfect_cube_l372_372439

/-- Given distinct prime numbers s, t, u, v. 
    Prove that the smallest positive perfect cube having m = st^2 u^3 v^5
    as a divisor is (stuv^2)^3.
-/ 
theorem smallest_perfect_cube (s t u v : ℕ)
  (hs : Prime s) (ht : Prime t) (hu : Prime u) (hv : Prime v)
  (hs_ne_ht : s ≠ t) (hs_ne_hu : s ≠ u) (hs_ne_hv : s ≠ v)
  (ht_ne_hu : t ≠ u) (ht_ne_hv : t ≠ v) (hu_ne_hv : u ≠ v) :
  ∃ n : ℕ, (m = s * t ^ 2 * u ^ 3 * v ^ 5) ∧ (n = s * t * u * v ^ 2) ^ 3 :=
by
  let m := s * t ^ 2 * u ^ 3 * v ^ 5
  use (s * t * u * v ^ 2)
  split
  · rfl
  · rfl

end smallest_perfect_cube_l372_372439


namespace division_quotient_l372_372834

theorem division_quotient (dividend divisor remainder quotient : ℕ) 
  (h₁ : dividend = 95) (h₂ : divisor = 15) (h₃ : remainder = 5)
  (h₄ : dividend = divisor * quotient + remainder) : quotient = 6 :=
by
  sorry

end division_quotient_l372_372834


namespace range_of_m_l372_372374

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4 * m - 5) * x^2 - 4 * (m - 1) * x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by
  sorry

end range_of_m_l372_372374


namespace altitude_foot_product_equality_l372_372810

theorem altitude_foot_product_equality (A B C H K L : Type) (hA : ∠ BAH = 90°) (hB : ∠ CBK = 90°) (hC : ∠ ACL = 90°) :
  (|AK| * |BL| * |CH| = |HK| * |KL| * |LH| ∧ |HK| * |KL| * |LH| = |AL| * |BH| * |CK| ) :=
by
  sorry

end altitude_foot_product_equality_l372_372810


namespace ellipse_a_value_l372_372025

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372025


namespace sum_even_ints_condition_l372_372771

theorem sum_even_ints_condition (a b : ℤ) (x y : ℤ) (n : ℤ) 
  (h_x : x = (n / 2) * (a + b)) 
  (h_y_even : (a % 2 = 0 ∧ b % 2 = 0) → y = n / 2) 
  (h_y_mixed : (a % 2 = 1 ∨ b % 2 = 1) → y = (n - 1) / 2)
  (h_n : n = b - a + 1)
  (h_sum : x + y = 611) :
  ∃ (a b : ℤ), x + y = 611 :=
by {
  sorry,
}

end sum_even_ints_condition_l372_372771


namespace area_of_third_region_l372_372585

theorem area_of_third_region (A B C : ℝ) 
    (hA : A = 24) 
    (hB : B = 13) 
    (hTotal : A + B + C = 48) : 
    C = 11 := 
by 
  sorry

end area_of_third_region_l372_372585


namespace second_year_students_numeric_methods_l372_372399

theorem second_year_students_numeric_methods :
  ∀ (A B : Finset ℕ), 
  (|A| = 450) → 
  (|A ∩ B| = 134) → 
  (|A ∪ B| = 541) → 
  (|B| = 225) :=
by
  intros A B hA hA_inter_B hA_union_B
  sorry

end second_year_students_numeric_methods_l372_372399


namespace increasing_function_condition_l372_372712

def f : ℝ → ℝ → ℝ :=
  λ a x, if x ≤ 1 then -x^2 - 2 * a * x - 5 else a / x

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-2 ≤ a ∧ a ≤ -1) :=
sorry

end increasing_function_condition_l372_372712


namespace Sara_snow_volume_l372_372127

theorem Sara_snow_volume :
  let length := 30
  let width := 3
  let first_half_length := length / 2
  let second_half_length := length / 2
  let depth1 := 0.5
  let depth2 := 1.0 / 3.0
  let volume1 := first_half_length * width * depth1
  let volume2 := second_half_length * width * depth2
  volume1 + volume2 = 37.5 :=
by
  sorry

end Sara_snow_volume_l372_372127


namespace roots_squared_sum_l372_372687

theorem roots_squared_sum :
  (∀ x, x^2 + 2 * x - 8 = 0 → (x = x1 ∨ x = x2)) →
  x1 + x2 = -2 ∧ x1 * x2 = -8 →
  x1^2 + x2^2 = 20 :=
by
  intros roots_eq_sum_prod_eq
  sorry

end roots_squared_sum_l372_372687


namespace currant_weight_l372_372591

noncomputable def volume_bucket : ℝ := 0.01 -- Volume of bucket in cubic meters
noncomputable def density_water : ℝ := 1000 -- Density of water in kg/m^3
noncomputable def packing_density : ℝ := 0.74 -- Packing density for currants

-- Effective volume occupied by currants
noncomputable def effective_volume : ℝ := volume_bucket * packing_density

-- Weight of the currants
def weight_of_currants : ℝ := density_water * effective_volume

theorem currant_weight :
  weight_of_currants = 7.4 :=
by
  sorry

end currant_weight_l372_372591


namespace total_days_on_island_correct_l372_372806

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end total_days_on_island_correct_l372_372806


namespace evaluate_expression_l372_372280

/-- 
Prove that the expression $\frac{1}{2-\sqrt{3}}-\pi^0-2\cos 30^{\circ}$ equals $1$ 
given the following conditions:
1. $\pi^0 = 1$
2. $\cos 30^{\circ} = \frac{\sqrt{3}}{2}$
3. Rationalizing $\frac{1}{2-\sqrt{3}}$: $\frac{1}{2-\sqrt{3}} \times \frac{2+\sqrt{3}}{2+\sqrt{3}} = 2 + \sqrt{3}$
-/

theorem evaluate_expression : 
  (1 / (2 - real.sqrt 3) - real.pi^0 - 2 * real.cos (real.pi / 6)) = 1 :=
by 
  have h1 : real.pi^0 = 1 := by sorry
  have h2 : real.cos (real.pi / 6) = real.sqrt 3 / 2 := by sorry
  have h3 : 1 / (2 - real.sqrt 3) = 2 + real.sqrt 3 := by sorry
  sorry

end evaluate_expression_l372_372280


namespace log_expression_eq_zero_l372_372925

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_eq_zero : 2 * log_base 5 10 + log_base 5 0.25 = 0 :=
by
  sorry

end log_expression_eq_zero_l372_372925


namespace probability_of_sum_16_with_modified_dice_l372_372236

def modified_dice_faces : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def total_outcomes (dice_faces : List ℕ) : ℕ := dice_faces.length * dice_faces.length

theorem probability_of_sum_16_with_modified_dice (dice_faces : List ℕ)
  (h_faces : dice_faces = modified_dice_faces)
  (h_total_outcomes : total_outcomes dice_faces = 64) :
  (let favorable_pairs := [(7, 9), (9, 7), (8, 8)] in
  (favorable_pairs.length : ℕ) / (total_outcomes dice_faces)) = 3 / 64 := 
by sorry

end probability_of_sum_16_with_modified_dice_l372_372236


namespace ellipse_foci_x_axis_l372_372827

theorem ellipse_foci_x_axis (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2)
    (h2 : ∀ x y : ℝ, (x^2 / sin α) + (y^2 / cos α) = 1) :
    (π / 4 < α) ∧ (α < π / 2) :=
sorry

end ellipse_foci_x_axis_l372_372827


namespace min_value_MN_l372_372334

section
variables {x y : ℝ}

/- Given conditions -/
def parabola_vertex := (0 : ℝ, 0 : ℝ)
def parabola_focus := (0 : ℝ, 1 : ℝ)

lemma parabola_equation : ∀ (x y : ℝ), y = x^2 / 4 ↔ (x, y) ∈ set_of (λ p: ℝ × ℝ, p.fst^2 = 4 * p.snd) :=
begin
  intros,
  split,
  { intro h,
    exact h.symm },
  { intro h,
    rw [h] },
end

theorem min_value_MN : ∃ (MN_min : ℝ), MN_min = 8 * sqrt 2 / 5 :=
begin
  use (8 * sqrt 2 / 5),
  sorry,
end

end min_value_MN_l372_372334


namespace percentage_left_handed_women_l372_372777

variables {x y : ℕ} (h1 : 3 * x + x = 4 * x) (h2 : 3 * y + 2 * y = 5 * y) (h3 : 4 * x = 5 * y)

theorem percentage_left_handed_women :
  (x / (3 * x + x) * 100) = 25 :=
begin
  -- Omitted proof here
  sorry
end

end percentage_left_handed_women_l372_372777


namespace find_a_l372_372020

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372020


namespace algorithm_finds_gcd_l372_372228

-- Mathematical equivalent problem: Prove that the given algorithm finds the GCD of x and y.

noncomputable def gcd_algorithm : ℕ → ℕ → ℕ
| x y := let m := max x y,
              n := min x y in
          if m % n = 0 then
            n
          else
            gcd_algorithm n (m % n)

theorem algorithm_finds_gcd (x y : ℕ) : gcd_algorithm x y = Nat.gcd x y := 
sorry

end algorithm_finds_gcd_l372_372228


namespace minimum_sum_distances_to_lines_l372_372354

open Real

def distance_point_to_line_y_eq (P : ℝ × ℝ) (k : ℝ) : ℝ := |P.2 - k|
def distance_point_to_line_x_eq (P : ℝ × ℝ) (k : ℝ) : ℝ := |P.1 + k|

theorem minimum_sum_distances_to_lines :
  (∀ P : ℝ × ℝ, (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 → 
    inf (distance_point_to_line_y_eq P 2 + distance_point_to_line_x_eq P 1) = 4 - sqrt 2) :=
sorry

end minimum_sum_distances_to_lines_l372_372354


namespace find_a9_l372_372352

variable {a : ℕ → ℝ} -- Define a as a sequence of real numbers

-- Define the conditions
def a1 := a 1 = 1 / 2
def cond2 := a 2 * a 8 = 2 * a 5 + 3
def geometric_sequence := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Lean 4 statement for the proof problem
theorem find_a9 (h_geo : geometric_sequence) (h_a1 : a1) (h_cond2 : cond2) : a 9 = 18 :=
by
  sorry

end find_a9_l372_372352


namespace min_M_value_l372_372129

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}
variable {M : ℕ}

def arithmetic_seq (n : ℕ) := 2 * n + 1
def geometric_seq (n : ℕ) := 2 ^ (n - 1)

noncomputable def T (n : ℕ) := ∑ i in finset.range n, (arithmetic_seq i / geometric_seq i)

theorem min_M_value : (∀ n : ℕ, T n < 10) → M ≥ 10 :=
sorry

end min_M_value_l372_372129


namespace find_a_l372_372021

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372021


namespace ellipse_eccentricity_a_l372_372009

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372009


namespace smallest_number_of_eggs_l372_372172

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372172


namespace inequality_solution_l372_372302

theorem inequality_solution (z : ℝ) : 
  z^2 - 40 * z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 :=
by
  sorry

end inequality_solution_l372_372302


namespace currant_weight_l372_372590

noncomputable def volume_bucket : ℝ := 0.01 -- Volume of bucket in cubic meters
noncomputable def density_water : ℝ := 1000 -- Density of water in kg/m^3
noncomputable def packing_density : ℝ := 0.74 -- Packing density for currants

-- Effective volume occupied by currants
noncomputable def effective_volume : ℝ := volume_bucket * packing_density

-- Weight of the currants
def weight_of_currants : ℝ := density_water * effective_volume

theorem currant_weight :
  weight_of_currants = 7.4 :=
by
  sorry

end currant_weight_l372_372590


namespace length_PQ_eq_external_tangent_l372_372790

-- Define basic properties and geometry of circles and tangents.
noncomputable def o : Type* := sorry
noncomputable def o' : Type* := sorry
noncomputable def P : o = o'
noncomputable def Q : o = o'
noncomputable def R : o = o'
noncomputable def S : o = o'
noncomputable def X : o = o'
noncomputable def Y : o = o'
noncomputable def V : o = o'
noncomputable def W : o = o'

-- Additional definitions for tangents
noncomputable def common_tangent_internal : o = o' := sorry
noncomputable def common_tangent_external : o = o' := sorry

-- PQ is defined as the length between intersection points of tangents
noncomputable def length_PQ : ℝ := sorry
noncomputable def length_external_tangent : ℝ := sorry

-- Theorem to be proved
theorem length_PQ_eq_external_tangent :
  (∀ P Q, P = common_tangent_internal -> Q = common_tangent_internal -> 
    length_PQ PQ = length_external_tangent) := by
  sorry

end length_PQ_eq_external_tangent_l372_372790


namespace compute_fraction_power_l372_372621

theorem compute_fraction_power (a b : ℕ) (ha : a = 123456) (hb : b = 41152) : (a ^ 5 / b ^ 5) = 243 := by
  sorry

end compute_fraction_power_l372_372621


namespace ellipse_a_value_l372_372027

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372027


namespace number_of_people_study_only_cooking_l372_372783

theorem number_of_people_study_only_cooking 
  (Y : ℕ) (Ck : ℕ) (W : ℕ) (CY : ℕ) (CW : ℕ) (C_Y_W : ℕ) 
  (H1 : Y = 25) 
  (H2 : Ck = 15) 
  (H3 : W = 8) 
  (H4 : CY = 7) 
  (H5 : C_Y_W = 3)
  (H6 : CW = 3) : (Ck - (CY - C_Y_W + CW - C_Y_W + C_Y_W) = 8) :=
by
  rw [H2, H4, H5, H6]
  sorry

end number_of_people_study_only_cooking_l372_372783


namespace smallest_positive_period_interval_of_monotonicity_cos_2x0_l372_372820

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x + π / 3) + sqrt 3 * (sin x ^ 2 - cos x ^ 2) - 1 / 2

theorem smallest_positive_period :
  ∀ x : ℝ, f (x + π) = f x := by
  sorry

theorem interval_of_monotonicity (k : ℤ) :
  ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → (∃ c, f (x + c) = f x) := by
  sorry

theorem cos_2x0 (x0 : ℝ) (hx0 : x0 ∈ set.Icc (5 * π / 12) (2 * π / 3)) (hf : f x0 = sqrt 3 / 3 - 1 / 2) :
  cos (2 * x0) = -(3 + sqrt 6) / 6 := by
  sorry

end smallest_positive_period_interval_of_monotonicity_cos_2x0_l372_372820


namespace ellipse_eccentricity_l372_372050

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372050


namespace hyperbola_center_l372_372936

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (3, 6)) (h₂ : (x2, y2) = (9, 10)) :
  midpoint (x1, y1) (x2, y2) = (6, 8) :=
by
  sorry

end hyperbola_center_l372_372936


namespace original_laborers_count_l372_372209

theorem original_laborers_count (L : ℕ) (h1 : (L - 7) * 10 = L * 6) : L = 18 :=
sorry

end original_laborers_count_l372_372209


namespace range_of_a_l372_372713

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then -x^2 - 2*a*x - 5 else a / x

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (is_increasing (f a) → a ∈ set.Icc (-2) (-1)) ∧
  (a ∈ set.Icc (-2) (-1) → is_increasing (f a)) :=
begin
  sorry
end

end range_of_a_l372_372713


namespace ratio_of_areas_l372_372425

def S_3 (x y : ℝ) := log (3 + x^2 + y^2) / log 10 ≤ 1 + log (x + y) / log 10
def S_4 (x y : ℝ) := log (4 + x^2 + y^2) / log 10 ≤ 2 + log (x + y) / log 10

theorem ratio_of_areas :
  let area_S_3 := π * 47
  let area_S_4 := π * 4996
  (area_S_4 / area_S_3) = 4996 / 47 :=
by
  sorry

end ratio_of_areas_l372_372425


namespace train_cross_pole_time_l372_372188

def km_per_hr_to_m_per_s (v_km_per_hr : ℕ) : ℕ :=
  v_km_per_hr * 1000 / 3600

theorem train_cross_pole_time
  (L : ℕ)
  (v_kmh : ℕ)
  (v_ms := km_per_hr_to_m_per_s v_kmh)
  (t := L / v_ms)
  (hL : L = 90)
  (hv : v_kmh = 72) :
  t = 4.5 := by
  sorry

end train_cross_pole_time_l372_372188


namespace interest_difference_l372_372919

def principal : ℝ := 1200
def rate : ℝ := 10 / 100
def time : ℝ := 1
def n : ℝ := 2   -- compounded half-yearly

def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t) - P

theorem interest_difference :
  compound_interest principal rate n time - simple_interest principal (rate * 100) time = 3 :=
by
  sorry

end interest_difference_l372_372919


namespace monomials_like_terms_l372_372376

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end monomials_like_terms_l372_372376


namespace complex_magnitude_example_l372_372743

theorem complex_magnitude_example (z : ℂ) (h : z = 1 + Complex.i) : Complex.abs (z^2 - 2*z) = 2 := 
by 
  rw [h]
  sorry

end complex_magnitude_example_l372_372743


namespace decimal_to_binary_87_l372_372629

theorem decimal_to_binary_87 :
  ∀ n, n = 87 → (λ bin, bin = 1010111) :=
begin
  assume n,
  assume hn : n = 87,
  sorry
end

end decimal_to_binary_87_l372_372629


namespace train_a_distance_traveled_l372_372920

variable (distance : ℕ) (speedA speedB : ℕ)

theorem train_a_distance_traveled 
  (h1 : distance = 450) 
  (h2 : speedA = 50) 
  (h3 : speedB = 50) :
  let combined_speed := speedA + speedB in
  let time := distance / combined_speed in
  time * speedA = 225 := by
  sorry

end train_a_distance_traveled_l372_372920


namespace geom_sequence_sum_of_first4_l372_372400

noncomputable def geom_sum_first4_terms (a : ℕ → ℝ) (common_ratio : ℝ) (a0 a1 a4 : ℝ) : ℝ :=
  a0 + a0 * common_ratio + a0 * common_ratio^2 + a0 * common_ratio^3

theorem geom_sequence_sum_of_first4 {a : ℕ → ℝ} (a1 a4 : ℝ) (r : ℝ)
  (h1 : a 1 = a1) (h4 : a 4 = a4) 
  (h_geom : ∀ n, a (n + 1) = a n * r) :
  geom_sum_first4_terms a (r) a1 (a 0) (a 4) = 120 :=
by sorry

end geom_sequence_sum_of_first4_l372_372400


namespace find_a_l372_372024

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372024


namespace distance_walked_in_hexagon_l372_372216

theorem distance_walked_in_hexagon
  (hexagon_side : ℝ)
  (walked_distance : ℝ)
  (start_point : ℝ × ℝ)
  (hexagon_side = 2)
  (walked_distance = 5)
  (start_point = (0, 0)) :
  let endpoint := (2.5, (3 * real.sqrt 3) / 2) in
  dist start_point endpoint = real.sqrt 13 := 
by
  let endpoint := (2.5, (3 * real.sqrt 3) / 2)
  have h : dist start_point endpoint = real.sqrt ((2.5 - 0) ^ 2 + (((3 * real.sqrt 3) / 2) - 0) ^ 2), by sorry
  rw [start_point, endpoint] at h
  exact h

end distance_walked_in_hexagon_l372_372216


namespace sum_f_8_to_2012_l372_372633

open Function

/-- Given function f: ℝ → ℝ which is continuous and even,
  and, when translated by 1 unit to the right, becomes an odd function. 
  And given f(2) = -1, prove that the finite sum of f(8) to f(2012) is 1. -/
theorem sum_f_8_to_2012 (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_right_translated_odd : ∀ x : ℝ, f (x + 1) = -f (-1 + x))
  (h_f2 : f 2 = -1) :
  (finset.sum (finset.range (2013 - 8)) (λ k, f (8 + k))) = 1 :=
sorry

end sum_f_8_to_2012_l372_372633


namespace general_term_formula_smallest_n_value_l372_372332

-- Define the sequence {a_n} with given conditions
def sequence (a : ℕ → ℕ) : Prop := ∀ n, n ≥ 2 → a(n) = 2 * a(n - 1)
def sum_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, S(n) = 2 * a(n) - a(1)
def arithmetic_sequence (a1 a2 : ℕ) : Prop := a1 + a2 = 2 * (a2 + 1)

-- Prove the general term of the sequence {a_n}
theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (a1 : ℕ) (a2 : ℕ) :
  sum_sequence S a →
  arithmetic_sequence a1 a2 →
  a(2) = 2 * a1 →
  a(3) = 4 * a1 →
  ∀ n, a(n) = 2 ^ n :=
sorry

-- Define the sum of the first n terms of the new sequence
def T_sequence (T : ℕ → ℝ) (a : ℕ → ℕ) : Prop := 
∀ n, T(n) = 1 - (1 / (2 ^ (n + 1) - 1))

-- Find the smallest value of n such that |T_n - 1| < 1/2016
theorem smallest_n_value (T : ℕ → ℝ) (a : ℕ → ℕ) :
  T_sequence T a →
  ∀ n, |T(n) - 1| < 1 / 2016 → n ≥ 10 :=
sorry

end general_term_formula_smallest_n_value_l372_372332


namespace ratio_of_democrats_l372_372143

variable (F M D_F D_M : ℕ)

theorem ratio_of_democrats (h1 : F + M = 750)
    (h2 : D_F = 1 / 2 * F)
    (h3 : D_F = 125)
    (h4 : D_M = 1 / 4 * M) :
    (D_F + D_M) / 750 = 1 / 3 :=
sorry

end ratio_of_democrats_l372_372143


namespace number_placement_l372_372156

theorem number_placement : 
  (∃ (squares : Fin 6 → ℕ), 
    (∀ i, squares i ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (Function.Injective squares) ∧ 
    (∀ i j, connected i j → (i > j → squares i > squares j))) 
  ↔ 
  (squares_placement_num_ways = 20) := 
sorry

end number_placement_l372_372156


namespace find_b_and_S_range_l372_372778

variable {A B C a b c : ℝ}
variables (S : ℝ)
variables (angleA_is_pi: A = π/4)
variables (sinAcosC_eq_3_cosAsinC: sin A * cos C = 3 * cos A * sin C)
variables (a2_min_c2_eq_2b: a ^ 2 - c ^ 2 = 2 * b)

theorem find_b_and_S_range :
  (b = 4) ∧
  (S + 8 * sqrt 2 * cos A * cos C ∈ Set.Ioc (-8 : ℝ) (8 * sqrt 2))
:=
sorry

end find_b_and_S_range_l372_372778


namespace number_of_ways_to_place_balls_l372_372463

theorem number_of_ways_to_place_balls : 
  let balls := 3 
  let boxes := 4 
  (boxes^balls = 64) :=
by
  sorry

end number_of_ways_to_place_balls_l372_372463


namespace annulus_area_l372_372232

-- Definitions and conditions from the problem
variables (R r l : ℝ) (h : R > r)

-- The Pythagorean theorem relationship given in the problem
def pythagorean_relation := R^2 = r^2 + l^2

-- The definition of the area of the annulus using the problem's conditions
def area_of_annulus := π * R^2 - π * r^2

-- The main theorem that transforms and simplifies the definition based on the earlier conditions
theorem annulus_area (h : R > r) (pythagorean_relation : R^2 = r^2 + l^2) :
  area_of_annulus R r = π * l^2 :=
by
  unfold area_of_annulus
  rw [pythagorean_relation]
  ring
  sorry

end annulus_area_l372_372232


namespace canada_population_l372_372776

theorem canada_population 
    (M : ℕ) (B : ℕ) (H : ℕ)
    (hM : M = 1000000)
    (hB : B = 2 * M)
    (hH : H = 19 * B) : 
    H = 38000000 := by
  sorry

end canada_population_l372_372776


namespace find_additional_discount_percentage_l372_372217

noncomputable def additional_discount_percentage(msrp : ℝ) (max_regular_discount : ℝ) (lowest_price : ℝ) : ℝ :=
  let regular_discount_price := msrp * (1 - max_regular_discount)
  let additional_discount := (regular_discount_price - lowest_price) / regular_discount_price
  additional_discount * 100

theorem find_additional_discount_percentage :
  additional_discount_percentage 40 0.3 22.4 = 20 :=
by
  unfold additional_discount_percentage
  simp
  sorry

end find_additional_discount_percentage_l372_372217


namespace point_in_third_quadrant_l372_372762

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := 
sorry

end point_in_third_quadrant_l372_372762


namespace largest_arithmetic_mean_two_digit_pairs_l372_372284

noncomputable def largest_arithmetic_mean : ℕ := 75

theorem largest_arithmetic_mean_two_digit_pairs :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a > b ∧
  (a + b) / 2 = (25 * (Int.natAbs ((Real.sqrt (a * b)).floor) + 1)) / 24 ∧
  ∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x > y ∧
                  (x + y) / 2 = (25 * (Int.natAbs ((Real.sqrt (x * y)).floor) + 1)) / 24 →
                  (x + y) / 2 ≤ largest_arithmetic_mean :=
begin
  sorry
end

end largest_arithmetic_mean_two_digit_pairs_l372_372284


namespace complex_sum_real_part_l372_372981

theorem complex_sum_real_part :
  (1 / 2 ^ 1988) * ∑ n in Finset.range 995, (-3 : ℂ) ^ n * (Nat.choose 1988 (2 * n) : ℂ) = -Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_sum_real_part_l372_372981


namespace min_height_of_box_with_surface_area_condition_l372_372604

theorem min_height_of_box_with_surface_area_condition {x : ℕ}  
(h : 2*x^2 + 4*x*(x + 6) ≥ 150) (hx: x ≥ 5) : (x + 6) = 11 := by
  sorry

end min_height_of_box_with_surface_area_condition_l372_372604


namespace independence_and_distributions_l372_372437

noncomputable def gamma_parameters 
  (α : ℕ → ℝ) (β : ℝ) (ξ : ℕ → ℝ) (i : ℕ) : Prop :=
  ∀ (i : ℕ), i ∈ {1, ..., n} → random_var ξ[i] ~ Gamma(α[i], β)

theorem independence_and_distributions {
  (ξ : ℕ → ℝ) (α : ℕ → ℝ) (β : ℝ) (n : ℕ) (hx : gamma_parameters α β ξ):
  ∀ i, i ∈ {1, ..., n-1} → 
    let ζ i := (ξ.sum(1, i+1))/(ξ.sum(1, i+2)) in
    let η := ξ.sum(0, n) in
    let X i := ξ i / ξ.sum(1, n) in
    Independece (ζ, η) ∧
    (η ~ Gamma(α.sum(0, n), β)) ∧
    (ζ i ~ Beta(α.sum(0, i), α.sum(0, i+1))) ∧
    (X ~ Dirichlet(α)) ∧
    ∀k, 1 ≤ k ≤ n-2 → 
    let Y i := X[i] / X.sum(1, k) in
    IsIndependence (Y, (X[(k+1)], ..., X[n-1])) ∧
    (Y ~ Dirichlet(α.sum(1, k)))
  :=
sorry

end independence_and_distributions_l372_372437


namespace element_in_set_l372_372229

def A := {p : ℚ × ℚ | ∃ k : ℤ, p.1 = (k / 3 : ℚ) ∧ p.2 = (k / 4 : ℚ)}

theorem element_in_set (x y : ℚ) (h : (x, y) = (4, 3)) :
  (x, y) ∈ A :=
by
  rw h
  use 12
  simp
  sorry

end element_in_set_l372_372229


namespace distance_between_roots_l372_372521

theorem distance_between_roots (a b : Fin 4016 → ℝ) (h_uniq_a : ∀ i j, i ≠ j → a i ≠ a j)
  (h_uniq_b : ∀ i j, i ≠ j → b i ≠ b j) (h_trinomials : ∀ i, (4 * b i ≤ a i ^ 2)) :
  ∃ x y, x ≠ y ∧ (dist x y ≤ 1/250) :=
begin
  -- sorry,
end

end distance_between_roots_l372_372521


namespace ellipse_eccentricity_l372_372049

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372049


namespace projection_on_Oxy_l372_372315

def vector := (ℝ × ℝ × ℝ)

def proj_xy (v : vector) : vector := (v.1, v.2, 0)

theorem projection_on_Oxy (a : vector) (h : a = (1, 2, -3)) : proj_xy a = (1, 2, 0) :=
by
  rw [h]
  rfl

end projection_on_Oxy_l372_372315


namespace tangent_circumcircle_l372_372794

open Real EuclideanGeometry

-- Declaration of points and angles in triangle ABC
variable {A B C M N K : Point}

variables (triangle_ABC : Triangle A B C)
variable (bisector_BK : Ray B K)
variable (on_BA : M ∈ LineSegment B A)
variable (on_BC : N ∈ LineSegment B C)
variable (angle_condition : ∡ A K M = ∡ C K N)
variable (half_angle_ABC : ∡ A K M = 1 / 2 * ∡ A B C)

theorem tangent_circumcircle (triangle_ABC : Triangle A B C)
  (bisector_BK : Ray B K)
  (on_BA : M ∈ LineSegment B A)
  (on_BC : N ∈ LineSegment B C)
  (angle_condition : ∡ A K M = ∡ C K N)
  (half_angle_ABC : ∡ A K M = 1 / 2 * ∡ A B C) :
  is_tangent (Circumcircle M B N) (Line A C) := by
  sorry

end tangent_circumcircle_l372_372794


namespace sum_prime_factors_of_3_pow_6_minus_1_l372_372998

theorem sum_prime_factors_of_3_pow_6_minus_1 :
  let n := 3^6 - 1 in
  sum (prime_factors n).to_finset = 22 :=
by
  let n := 3^6 - 1
  have H : n = 28 * 26 :=
    calc
      n = (3^3 + 1) * (3^3 - 1) : by ring
      ... = 28 * 26 : by norm_num
  have F1 : prime_factors 28 = [2, 2, 7] := by norm_num
  have F2 : prime_factors 26 = [2, 13] := by norm_num
  have P : prime_factors n = [2, 2, 2, 7, 13] := by simp [prime_factors, H, F1, F2]
  have : (prime_factors n).to_finset = {2, 7, 13} := by simp [P]
  have : sum (prime_factors n).to_finset = 2 + 7 + 13 := by simp
  exact this

end sum_prime_factors_of_3_pow_6_minus_1_l372_372998


namespace circumcenter_on_circumcircle_of_triangle_ABC_l372_372428

-- Define the geometric setup: circle S1, points A, B, C, D.
variables (S₁ S₂ : Type) [circle S₁] [circle S₂]
          (A B C D : point) (tangent_line : line)
          (A_tangent : on_tangent_line A B tangent_line S₁) 
          (C_segment : on_line_segment C A S₁)
          (S₂_touching : is_touching S₂ AC C S₁ D) 
          (circumcenter_BCD : point)

-- Given B is a point on circle S₁
-- Given A is a point on tangent at B to S₁
-- Given C is a point not on S₁ such that the line segment AC meets S₁ at two distinct points
-- Given S₂ touches AC at C and S₁ at D on the opposite side of AC from B
-- Prove that the circumcenter of triangle BCD lies on the circumcircle of triangle ABC

theorem circumcenter_on_circumcircle_of_triangle_ABC (circumcenter_BCD_on_circumcircle_ABC : circumcenter_triangle B C D circumcenter_BCD)
        (S₁_circumcircle_ABC: circumcircle_triangle A B C S₁) : 
        is_on_circumcircle circumcenter_BCD (circumcircle A B C) :=
sorry

end circumcenter_on_circumcircle_of_triangle_ABC_l372_372428


namespace shadow_change_sequence_l372_372168

-- Define the height of the person and the height of the street lamp.
variables (h_person h_lamp : ℝ) (H_height : h_person < h_lamp)

-- Define the distance function for when a person is far from, under, and moving away
def shadow_length := λ d : ℝ, if d = 0 then h_person else ((h_lamp - h_person) / d) * h_person

-- Define conditions for person being 'far from', 'under', and 'away from' the street lamp
def far_from_lamp (d : ℝ) : Prop := d > 1
def under_lamp (d : ℝ) : Prop := d = 0
def away_from_lamp (d : ℝ) : Prop := d < 1 ∧ d > 0

-- Define the sequence of shadow length changes as a person walks under a street lamp.
theorem shadow_change_sequence :
  ∀ d : ℝ, (far_from_lamp d → shadow_length d > h_person) ∧
           (under_lamp d → shadow_length d = h_person) ∧
           (away_from_lamp d → shadow_length d > h_person) :=
by 
  sorry

end shadow_change_sequence_l372_372168


namespace Jaime_saves_27_per_week_l372_372275

-- Define the weekly saving amount as x
variables {x : ℝ} {savings : ℝ} 

-- Given conditions
def condition1 : Prop := 
  (5 : ℝ) * x = 135

-- Prove that Jaime saves $27 each week
theorem Jaime_saves_27_per_week (h : condition1) : x = 27 :=
by { sorry }

end Jaime_saves_27_per_week_l372_372275


namespace find_number_l372_372461

theorem find_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 14) : 0.40 * N = 168 :=
sorry

end find_number_l372_372461


namespace intersection_points_count_l372_372877

-- Definitions of the logarithmic functions
def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

-- Given functions
def f1 (x : ℝ) := log_base 2 x
def f2 (x : ℝ) := log_base x 2
def f3 (x : ℝ) := log_base (1 / 2) x
def f4 (x : ℝ) := log_base x (1 / 2)

-- Lean 4 math proof problem
theorem intersection_points_count :
  { x : ℝ | x > 0 ∧ (∃ y : ℝ, (f1 x = y ∨ f2 x = y ∨ f3 x = y ∨ f4 x = y) ∧
                          (∃ z : ℝ, z ≠ y ∧ (f1 x = z ∨ f2 x = z ∨ f3 x = z ∨ f4 x = z)))
  }.to_finset.card = 3 :=
by
  sorry

end intersection_points_count_l372_372877


namespace problem1_solution_problem2_solution_l372_372296

noncomputable def problem1_expr : ℝ :=
  2^(Real.log (1/4)/Real.log (2)) + (64/27)^(-1/3) + Real.log 5 3 * (Real.log 3 10 - Real.log 3 2)

theorem problem1_solution : problem1_expr = 2 := by
  sorry

noncomputable def problem2_expr : ℝ :=
  8^(-1/3) + Real.log 3 (1/27) + Real.log 6 5 * (Real.log 5 2 + Real.log 5 3) + 10^(Real.log 10 3)

theorem problem2_solution : problem2_expr = 3/2 := by
  sorry

end problem1_solution_problem2_solution_l372_372296


namespace PQT_is_isosceles_l372_372923

theorem PQT_is_isosceles {A B C D E P Q T : Point} 
  (h_cyclic : CyclicPentagon A B C D E)
  (h_convex : Convex A B C D E)
  (h_AB_BC : AB = BC)
  (h_CD_DE : CD = DE)
  (h_AD_BE_P : LinesIntersectAt AD BE P)
  (h_BD_CA_Q : LinesIntersectAt BD CA Q)
  (h_BD_CE_T : LinesIntersectAt BD CE T) :
  IsoscelesTriangle P Q T := 
sorry

end PQT_is_isosceles_l372_372923


namespace circle_equation_l372_372699

theorem circle_equation (a r : ℝ) (h1 : ∀ y : ℝ, (0, y) = (0, -y)) (h2 : ∃ x y : ℝ, x = 1 ∧ y = 0) (h3 : r = sqrt (4 / 3)) :
  ∃ a : ℝ, (a = sqrt (3) / 3 ∨ a = -sqrt (3) / 3) ∧ ∀ x y : ℝ, x^2 + (y - a)^2 = r :=
by
  sorry

end circle_equation_l372_372699


namespace ashu_job_completion_l372_372103

theorem ashu_job_completion :
  (Suresh_time Ashutosh_time : ℝ)
  (hSuresh : Suresh_time = 15) 
  (hAshu_remaining : ∀ Suresh_work_time remaining Ashu_work_time : ℝ,
    Suresh_work_time = 9 → remaining = 1 - (Suresh_work_time / Suresh_time) → Ashu_work_time = 10 → 
    (remaining / Ashu_work_time) = 1 / 25) :
  Ashutosh_time = 25 := 
begin
  have hS : Suresh_time = 15 := hSuresh,
  have hA : (1 - (9 / Suresh_time)) / 10 = 1 / 25 := hAshu_remaining 9 (1 - (9 / 15)) 10 rfl rfl rfl,
  simp at hA,
  linarith only [hS, hA],
end

end ashu_job_completion_l372_372103


namespace inlet_rate_correct_l372_372211

noncomputable def leak_empty_time : ℝ := 6 -- hours
noncomputable def both_empty_time : ℝ := 8 -- hours
noncomputable def tank_capacity : ℝ := 5760 -- liters
noncomputable def inlet_rate_liters_per_minute : ℝ :=
  (let leak_rate := tank_capacity / leak_empty_time in
   let both_rate := tank_capacity / both_empty_time in
   let inlet_rate := both_rate + leak_rate in
   inlet_rate / 60)

theorem inlet_rate_correct :
  inlet_rate_liters_per_minute = 28 := by
  sorry

end inlet_rate_correct_l372_372211


namespace inequality1_inequality2_l372_372099

theorem inequality1 (x : ℝ) : 
  x^2 - 2 * x - 1 > 0 -> x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1 := 
by sorry

theorem inequality2 (x : ℝ) : 
  (2 * x - 1) / (x - 3) ≥ 3 -> 3 < x ∧ x <= 8 := 
by sorry

end inequality1_inequality2_l372_372099


namespace auntie_em_can_park_l372_372940

-- Definition of the problem conditions
def parking_lot_spaces := 18
def required_car_spaces := 12
def suv_spaces := 2
def remaining_spaces := parking_lot_spaces - required_car_spaces

-- Definition of the solution: number of ways to choose 6 spaces (remaining spaces)
def total_ways_to_choose_spaces := (Finset.image (λ x : Fin 18, 18.choose x) Finset.univ).card

-- Definition of unfavorable configurations where no two empty spaces are adjacent
def unfavorable_ways_to_choose_spaces := (Finset.image (λ x : Fin 13, 13.choose x) Finset.univ).card

-- Definition of the ratio
def probability_not_able_to_park := (unfavorable_ways_to_choose_spaces: ℚ) / (total_ways_to_choose_spaces: ℚ)
def probability_able_to_park := 1 - probability_not_able_to_park

-- Proof of the given probability
theorem auntie_em_can_park : probability_able_to_park = 1403 / 1546 := by
  sorry

end auntie_em_can_park_l372_372940


namespace ellipse_eccentricity_l372_372038

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372038


namespace cube_sphere_volume_ratio_l372_372690

theorem cube_sphere_volume_ratio (a r : ℝ) (h : 6 * a^2 = 4 * Real.pi * r^2) :
  (a^3) / ((4 / 3) * Real.pi * r^3) = Real.sqrt(6 * Real.pi) / 6 :=
sorry

end cube_sphere_volume_ratio_l372_372690


namespace eval_expression_l372_372614

theorem eval_expression : 
  real.cbrt 8 + (1 / 3)⁻¹ - 2 * real.cos (real.pi / 6) + |1 - real.sqrt 3| = 4 :=
by 
  have h1: real.cbrt 8 = 2 := by norm_num,
  have h2: (1 / 3)⁻¹ = 3 := by norm_num,
  have h3: 2 * real.cos (real.pi / 6) = real.sqrt 3 := by norm_num,
  have h4: |1 - real.sqrt 3| = real.sqrt 3 - 1 := 
    by { rw real.abs_of_nonpos (sub_nonpos.mpr (real.sqrt_lt_sqrt (show (1:ℝ)^2 < (real.sqrt 3)^2, by norm_num))) },
  sorry

end eval_expression_l372_372614


namespace smallest_radius_of_tangent_sphere_l372_372934

theorem smallest_radius_of_tangent_sphere:
  ∃ R : ℝ,
  let side_length_of_large_cube := 3 in
  let side_length_of_unit_cube := 1 in
  let radius_of_corner_sphere := side_length_of_unit_cube / 2 in
  let distance_from_center_of_large_cube_to_corner := (side_length_of_large_cube * real.sqrt 3) / 2 in
  let radius_of_last_sphere := distance_from_center_of_large_cube_to_corner - radius_of_corner_sphere - radius_of_corner_sphere in
  R = radius_of_last_sphere ∧ R = (3 * real.sqrt 3 / 2) - 1 :=
begin
  sorry
end

end smallest_radius_of_tangent_sphere_l372_372934


namespace average_one_eighth_one_sixth_l372_372108

theorem average_one_eighth_one_sixth :
  (1/8 + 1/6) / 2 = 7/48 := 
by
  sorry

end average_one_eighth_one_sixth_l372_372108


namespace even_three_digit_numbers_count_l372_372155

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def valid_hundreds : Finset ℕ := {d ∈ digits | d < 3}  -- {1, 2}
def valid_tens : Finset ℕ := digits                     -- {1, 2, 3, 4, 5}
def valid_units : Finset ℕ := {d ∈ digits | d % 2 = 0}  -- {2, 4}

theorem even_three_digit_numbers_count : 
  (valid_hundreds.card * valid_tens.card * valid_units.card) = 20 := by
  sorry

end even_three_digit_numbers_count_l372_372155


namespace simplify_expression_correct_l372_372479

noncomputable def simplify_expression : Prop :=
  (1 / (Real.log 3 / Real.log 6 + 1) + 1 / (Real.log 7 / Real.log 15 + 1) + 1 / (Real.log 4 / Real.log 12 + 1)) = -Real.log 84 / Real.log 10

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end simplify_expression_correct_l372_372479


namespace slopes_of_asymptotes_l372_372269

theorem slopes_of_asymptotes (a b : ℝ) (h₁ : a^2 = 16) (h₂ : b^2 = 9) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -(3 / 4) :=
by
  have ha : a = 4, from sorry,
  have hb : b = 3, from sorry,
  use (b / a)
  split
  · exact sorry
  · exact sorry

end slopes_of_asymptotes_l372_372269


namespace probability_of_usable_parts_l372_372954

def isUsable (x y : ℝ) : Prop := 
  x >= 50 ∧ y >= 50 ∧ 200 - x - y >= 50

def totalArea : ℝ := 20000

def usableArea : ℝ := 1250

theorem probability_of_usable_parts : 
  (∫ x in 0..200, ∫ y in 0..200 - x, if isUsable x y then 1 else 0) / totalArea = (1 / 16) :=
by
  sorry

end probability_of_usable_parts_l372_372954


namespace three_digit_numbers_with_4_and_5_l372_372366

theorem three_digit_numbers_with_4_and_5 : 
  (card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ d1 d2 d3 : ℕ, d1 * 100 + d2 * 10 + d3 = n ∧ (d1 = 4 ∨ d2 = 4 ∨ d3 = 4) ∧ (d1 = 5 ∨ d2 = 5 ∨ d3 = 5) }) = 48 :=
by
  sorry

end three_digit_numbers_with_4_and_5_l372_372366


namespace angle_BLC_twice_angle_BAC_l372_372316

-- Definitions for the given conditions
variables {A B C K L : Point}
variables [triangle : Triangle A B C]
variables [circumcircle : Circumcircle A B C]
variables [symmedian_line : SymmedianLine A (Median A B C)]
variables [midpoint_L : Midpoint L A K]

-- Statement of the theorem
theorem angle_BLC_twice_angle_BAC 
  (h1 : symmedian_line.intersection_circumcircle = (A, K))
  (h2 : Midpoint L A K) :
  ∠BLC = 2 * ∠BAC :=
begin 
  sorry
end

end angle_BLC_twice_angle_BAC_l372_372316


namespace find_time_due_l372_372494

noncomputable def bank_time_in_years
  (BG : ℝ) (PW : ℝ) (Rate : ℝ) : ℝ :=
let T := (5 : ℝ) / 12 in 
BG = (PW * (Rate / 100) * T) / (1 + (Rate / 100) * T)

theorem find_time_due
  (BG : ℝ) (PW : ℝ) (Rate : ℝ) (h_BG : BG = 24) (h_PW : PW = 600) (h_Rate : Rate = 10) :
  bank_time_in_years BG PW Rate :=
by
  sorry

end find_time_due_l372_372494


namespace cone_to_prism_volume_ratio_l372_372587

theorem cone_to_prism_volume_ratio (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let V_cone := (1 / 3) * π * (w / 2)^2 * h,
      V_prism := 2 * w^2 * h in
  V_cone / V_prism = π / 24 :=
by
  let r := w / 2,
  let V_cone := (1 / 3) * π * r^2 * h,
  let V_prism := 2 * w^2 * h,
  have h_cone_vol : V_cone = (1 / 12) * π * w^2 * h := by
    rw [←mul_assoc, mul_comm (1 / 3 : ℝ), ←mul_assoc, mul_comm r r],
    norm_num,
    ring,
  have h_prism_vol : V_prism = 2 * w^2 * h := rfl,
  have h_ratio := (V_cone / V_prism : ℝ),
  rw [h_cone_vol, h_prism_vol, div_eq_mul_inv, ←mul_assoc, mul_comm h V_prism],
  norm_num,
  ring,
  sorry

end cone_to_prism_volume_ratio_l372_372587


namespace triangle_sides_abs_diff_l372_372326

theorem triangle_sides_abs_diff {a b c : ℝ} (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  |a - b + c| - |c - a - b| = 2c - 2b :=
by
  -- Skipping the proof as instructed
  sorry

end triangle_sides_abs_diff_l372_372326


namespace probability_B_winning_l372_372855

def P_A : ℝ := 0.2
def P_D : ℝ := 0.5
def P_B : ℝ := 1 - (P_A + P_D)

theorem probability_B_winning : P_B = 0.3 :=
by
  -- Proof steps go here
  sorry

end probability_B_winning_l372_372855


namespace train_speed_l372_372953

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 630) (h_time : time = 36) :
  (length / 1000) / (time / 3600) = 63 :=
by
  rw [h_length, h_time]
  sorry

end train_speed_l372_372953


namespace ellipse_a_value_l372_372028

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372028


namespace fifth_selected_individual_l372_372594

-- Representation of the set and selection procedure
setOption quotPrecheck false

theorem fifth_selected_individual {n : ℕ} (num_table : list (list (nat))) :
  let individuals := [1, 2, ..., 20] in
  let first_row := num_table.head in
  let cols_five_six := first_row.drop 4 in
  let select_numbers := (cols_five_six.head.digits.take 2 ++ cols_five_six.tail.head.digits.take 2).filter (λ x, x < 20) in
  let extended_selection := select_numbers ++ [8, 2, 14, 7, 1] in
  List.nth extended_selection 4 = 1 := 
by
  sorry

end fifth_selected_individual_l372_372594


namespace process_tents_l372_372233

/-- Problem statement: An earthquake occurred in a certain area, 
and 550 tents are needed to solve the temporary accommodation problem 
for the disaster-affected people. 
Now two factories, A and B, are processing and producing the tents. 
It is known that the daily processing capacity of factory A is 1.5 times that of factory B, 
and factory A takes 4 days less than factory B to process and produce 240 tents.

1. Find out how many tents each factory can process and produce per day.
2. If the daily processing cost of factory A is 30,000 yuan and 
the daily processing cost of factory B is 24,000 yuan, 
in order to ensure that the total processing cost of these disaster relief tents 
does not exceed 600,000 yuan, how many days should factory A be arranged to process and produce at least.
-/

noncomputable def daily_processing_capacity (x : ℕ) (y : ℕ) :=
  x = 20 ∧ y = 30

noncomputable def min_processing_days (d : ℕ) :=
  d ≥ 10

theorem process_tents
  (x y : ℕ)
  (hx : daily_processing_capacity x y) :
  min_processing_days 10 :=
begin
  sorry
end

end process_tents_l372_372233


namespace mari_vs_kendra_l372_372074

-- Variable Definitions
variables (K M S : ℕ)  -- Number of buttons Kendra, Mari, and Sue made
variables (h1: 2*S = K) -- Sue made half as many as Kendra
variables (h2: S = 6)   -- Sue made 6 buttons
variables (h3: M = 64)  -- Mari made 64 buttons

-- Theorem Statement
theorem mari_vs_kendra (K M S : ℕ) (h1 : 2 * S = K) (h2 : S = 6) (h3 : M = 64) :
  M = 5 * K + 4 :=
sorry

end mari_vs_kendra_l372_372074


namespace line_not_in_first_quadrant_l372_372767

theorem line_not_in_first_quadrant (t : ℝ) : 
  (∀ x y : ℝ, ¬ ((0 < x ∧ 0 < y) ∧ (2 * t - 3) * x + y + 6 = 0)) ↔ t ≥ 3 / 2 :=
by
  sorry

end line_not_in_first_quadrant_l372_372767


namespace smallest_m_proof_l372_372225

noncomputable def smallest_m := 53

theorem smallest_m_proof : 
  ∃ m : ℕ, (∀ x : ℕ, 1.06 * x = m * 100 → m = smallest_m) :=
by
  sorry

end smallest_m_proof_l372_372225


namespace product_of_solutions_of_t_squared_eq_49_l372_372668

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372668


namespace ellipse_a_value_l372_372026

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372026


namespace seq_bounded_seq_convergent_l372_372401

noncomputable def seq (n : ℕ) : ℝ :=
if n = 1 then 1 else (√2)^(seq (n-1))

theorem seq_bounded :
  ∃ M, ∀ n, seq n ≤ M := 
sorry

theorem seq_convergent :
  ∃ l, tendsto seq at_top (𝓝 l) := 
sorry

end seq_bounded_seq_convergent_l372_372401


namespace sum_faces_edges_vertices_square_pyramid_l372_372543

theorem sum_faces_edges_vertices_square_pyramid : 
  let F := 5 in
  let E := 8 in
  let V := 5 in
  F + E + V = 18 := by
  sorry

end sum_faces_edges_vertices_square_pyramid_l372_372543


namespace matrix_product_is_correct_l372_372620

-- Define matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, -1, 2],
  ![4, 5, -3],
  ![1, 1, 0]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![0, 6, -1],
  ![-2, 3, 4],
  ![5, -1, 2]
]

-- Define the result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![12, 13, -3],
  ![-25, 42, 10],
  ![-2, 9, 3]
]

-- The theorem stating the product of A and B is C
theorem matrix_product_is_correct : A ⬝ B = C := by
  sorry

end matrix_product_is_correct_l372_372620


namespace increasing_function_condition_l372_372711

def f : ℝ → ℝ → ℝ :=
  λ a x, if x ≤ 1 then -x^2 - 2 * a * x - 5 else a / x

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-2 ≤ a ∧ a ≤ -1) :=
sorry

end increasing_function_condition_l372_372711


namespace ellipse_eccentricity_l372_372039

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372039


namespace grasshopper_jump_is_31_l372_372879

def frog_jump : ℕ := 35
def total_jump : ℕ := 66
def grasshopper_jump := total_jump - frog_jump

theorem grasshopper_jump_is_31 : grasshopper_jump = 31 := 
by
  unfold grasshopper_jump
  sorry

end grasshopper_jump_is_31_l372_372879


namespace abs_z_squared_minus_two_z_eq_two_l372_372747

theorem abs_z_squared_minus_two_z_eq_two (z : ℂ) (hz : z = 1 + 1*Complex.i) : 
  |z^2 - 2*z| = 2 :=
begin
  sorry
end

end abs_z_squared_minus_two_z_eq_two_l372_372747


namespace ellipse_eccentricity_l372_372042

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372042


namespace corrected_line_segment_length_l372_372581

-- Define the constants based on the conditions
def scale_factor_inch_to_feet : ℝ := 500 / 2
def corrected_length_in_inches : ℝ := 7.25

-- Define the problem statement
theorem corrected_line_segment_length :
  corrected_length_in_inches * scale_factor_inch_to_feet = 1812.5 :=
by
  -- Proof will go here
  sorry

end corrected_line_segment_length_l372_372581


namespace part1_part2_l372_372829

universe u

def A := {x : ℝ | 2 * x ^ 2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) := {x : ℝ | x ^ 2 + a < 0}
def complement (S : set ℝ) := {x : ℝ | x ∉ S}

theorem part1 (a : ℝ) (ha : a = -4) : 
  (A ∩ B a) = {x : ℝ | -2 < x ∧ x ≤ 3} ∧ 
  (A ∪ B a) = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by sorry

theorem part2 (a : ℝ) (ha_neg : a < 0) (H : (complement A ∩ B a) = B a) :
  -1 / 4 ≤ a ∧ a < 0 :=
by sorry

end part1_part2_l372_372829


namespace max_value_z_conj_z_l372_372765

noncomputable def Z (θ : ℝ) : ℂ := 2 * complex.cos θ + complex.i * complex.sin θ

theorem max_value_z_conj_z (θ : ℝ) : ∃ θ : ℝ, (complex.norm_sq (Z θ)) = 4 :=
by
  sorry

end max_value_z_conj_z_l372_372765


namespace increasing_function_on_R_l372_372468

theorem increasing_function_on_R (x1 x2 : ℝ) (h : x1 < x2) : 3 * x1 + 2 < 3 * x2 + 2 := 
by
  sorry

end increasing_function_on_R_l372_372468


namespace change_correctness_l372_372380

theorem change_correctness (cost_milk cost_water given_amount : ℕ) (h_milk : cost_milk = 350) (h_water : cost_water = 500) (h_amount : given_amount = 1000) : 
  given_amount - (cost_milk + cost_water) = 150 :=
by
  -- Steps of the proof
  rw [h_milk, h_water, h_amount]
  norm_num
  apply sorry

end change_correctness_l372_372380


namespace cyclist_motorcyclist_speed_l372_372115

def speed_of_motorcyclist (x : ℝ) : Prop :=
  let speed_cyclist := x - 30 in
  let time_motorcyclist := 120 / x in
  let time_cyclist := 120 / speed_cyclist in
  time_cyclist = time_motorcyclist + 2

theorem cyclist_motorcyclist_speed :
  ∃ x : ℝ, (x > 0) ∧ (x - 30 > 0) ∧ speed_of_motorcyclist x :=
  sorry

end cyclist_motorcyclist_speed_l372_372115


namespace ellipse_eccentricity_l372_372051

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372051


namespace rate_of_painting_l372_372507

def length : ℝ := 23
def breadth : ℝ := length / 3
def total_cost : ℝ := 529
def area : ℝ := length * breadth
def rate : ℝ := total_cost / area

theorem rate_of_painting : rate ≈ 3 := by
  sorry -- Proof is not required

end rate_of_painting_l372_372507


namespace geometric_progression_properties_l372_372500

-- Define the first term and the fifth term given
def b₁ := Real.sqrt 3
def b₅ := Real.sqrt 243

-- Define the nth term formula for geometric progression
def geometric_term (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

-- State both the common ratio and the sixth term
theorem geometric_progression_properties :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ 
           geometric_term b₁ q 5 = b₅ ∧ 
           geometric_term b₁ q 6 = 27 ∨ geometric_term b₁ q 6 = -27 :=
by
  sorry

end geometric_progression_properties_l372_372500


namespace ellipse_a_value_l372_372032

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372032


namespace complex_conjugate_example_l372_372766

open Complex

theorem complex_conjugate_example :
  let z := i * (3 - 2 * i)
  conj z = 2 - 3 * i :=
by
  let z := i * (3 - 2 * i)
  sorry

end complex_conjugate_example_l372_372766


namespace Gwen_deleted_pictures_l372_372553

theorem Gwen_deleted_pictures :
  ∀ (pictures_zoo pictures_museum remaining_pictures : ℕ),
  pictures_zoo = 41 →
  pictures_museum = 29 →
  remaining_pictures = 55 →
  (pictures_zoo + pictures_museum - remaining_pictures) = 15 :=
by
  intros pictures_zoo pictures_museum remaining_pictures
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Gwen_deleted_pictures_l372_372553


namespace max_xy_l372_372377

-- Lean statement for the given problem
theorem max_xy (x y : ℝ) (h : x^2 + y^2 = 4) : xy ≤ 2 := sorry

end max_xy_l372_372377


namespace lois_purchased_books_l372_372073

-- Defining initial condition
def initial_books : ℕ := 40

-- Defining the books given to nephew and left after
def books_given_to_nephew (total : ℕ) : ℕ := total / 4
def books_remaining_after_nephew (total : ℕ) : ℕ := total - books_given_to_nephew total

-- Defining the books given to the library and left after
def books_given_to_library (remaining : ℕ) : ℕ := remaining / 3
def books_remaining_after_library (remaining : ℕ) : ℕ := remaining - books_given_to_library remaining

-- Final known condition
def final_books : ℕ := 23

-- Theorem to prove Lois purchased 3 books 
theorem lois_purchased_books :
  let total_initial := initial_books,
      after_nephew := books_remaining_after_nephew total_initial,
      after_library := books_remaining_after_library after_nephew in
  final_books - after_library = 3 :=
by sorry

end lois_purchased_books_l372_372073


namespace product_of_t_values_l372_372662

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372662


namespace money_inequalities_l372_372297

theorem money_inequalities (a b : ℝ) (h₁ : 5 * a + b > 51) (h₂ : 3 * a - b = 21) : a > 9 ∧ b > 6 := 
by
  sorry

end money_inequalities_l372_372297


namespace largest_constant_l372_372536

def equation_constant (c d : ℝ) : ℝ :=
  5 * c + (d - 12)^2

theorem largest_constant : ∃ constant : ℝ, (∀ c, c ≤ 47) → (∀ d, equation_constant 47 d = constant) → constant = 235 := 
by
  sorry

end largest_constant_l372_372536


namespace ellipse_eccentricity_l372_372037

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372037


namespace gcd_960_1632_l372_372646

theorem gcd_960_1632 : Int.gcd 960 1632 = 96 := by
  sorry

end gcd_960_1632_l372_372646


namespace total_days_on_island_correct_l372_372805

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end total_days_on_island_correct_l372_372805


namespace circle_symmetry_properties_l372_372206

theorem circle_symmetry_properties (C : Type*) [metric_space C] (circle : {c : set C // ∃ p r, c = metric.sphere p r}) :
  (∀ p r, circle.val = metric.sphere p r → is_axial_symmetric (metric.sphere p r)) ∧
  (∀ p r l, circle.val = metric.sphere p r → metric.line_through p r l → l ∈ axis_of_symmetry (metric.sphere p r)) ∧
  (∀ p r, circle.val = metric.sphere p r → infinite_axes_of_symmetry (metric.sphere p r)) :=
by
  sorry

end circle_symmetry_properties_l372_372206


namespace yan_distance_ratio_l372_372969

theorem yan_distance_ratio 
  (w x y : ℝ)
  (h1 : y / w = x / w + (x + y) / (10 * w)) :
  x / y = 9 / 11 :=
by
  sorry

end yan_distance_ratio_l372_372969


namespace cookies_left_for_Monica_l372_372455

-- Definitions based on the conditions
def total_cookies : ℕ := 30
def father_cookies : ℕ := 10
def mother_cookies : ℕ := father_cookies / 2
def brother_cookies : ℕ := mother_cookies + 2

-- Statement for the theorem
theorem cookies_left_for_Monica : total_cookies - (father_cookies + mother_cookies + brother_cookies) = 8 := by
  -- The proof goes here
  sorry

end cookies_left_for_Monica_l372_372455


namespace round_to_nearest_hundredth_l372_372238

theorem round_to_nearest_hundredth (x : ℝ) (hx : x = 3.1415926) : Real.round_to (10 ^ (-2 : ℤ)) x = 3.14 :=
by
  -- We include this "sorry" to skip the proof as instructed.
  sorry

end round_to_nearest_hundredth_l372_372238


namespace sum_fractional_parts_arithmetic_progression_l372_372318

noncomputable def fractional_part (x : ℝ) : ℝ := x - (⌊x⌋ : ℝ)

theorem sum_fractional_parts_arithmetic_progression :
  let a_1 := 9.5 - 2 * 0.6 in
  let a (n : ℕ) := a_1 + (n - 1) * 0.6 in
  (∑ n in finset.range 100, fractional_part (a (n + 1))) = 50 :=
by
  let a_1 := 9.5 - 2 * 0.6
  let a (n : ℕ) := a_1 + (n - 1) * 0.6
  have := (∑ n in finset.range 100, fractional_part (a (n + 1))) = 50
  sorry 

end sum_fractional_parts_arithmetic_progression_l372_372318


namespace charlie_floor_ratio_l372_372264

-- Let's define the conditions as constants or variables
variables (Dennis Charlie Frank : ℕ)

-- Conditions defined
axiom Dennis_floor : Dennis = 6
axiom Frank_floor : Frank = 16
axiom Dennis_above_Charlie : Dennis = Charlie + 2

-- The theorem statement we need to prove
theorem charlie_floor_ratio (H1 : Dennis = 6) (H2 : Frank = 16) (H3 : Dennis = Charlie + 2) :
  (Charlie / Frank) = 1 / 4 :=
begin
  sorry
end

end charlie_floor_ratio_l372_372264


namespace solve_quadratic_1_solve_quadratic_2_l372_372851

theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x - 1 = 0 ↔ x = 1 + real.sqrt 2 ∨ x = 1 - real.sqrt 2 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : 3*x*(x - 1) = 2*x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l372_372851


namespace y_gt_1_l372_372124

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end y_gt_1_l372_372124


namespace problem_solution_l372_372976

noncomputable def complex_expression : ℝ :=
  (-(1/2) * (1/100))^5 * ((2/3) * (2/100))^4 * (-(3/4) * (3/100))^3 * ((4/5) * (4/100))^2 * (-(5/6) * (5/100)) * 10^30

theorem problem_solution : complex_expression = -48 :=
by
  sorry

end problem_solution_l372_372976


namespace chromium_percentage_l372_372918

noncomputable section

/-- Definitions for the conditions -/
def amount_chromium_first_alloy := 0.12 * 15
def amount_chromium_second_alloy := 0.08 * 40
def total_chromium := amount_chromium_first_alloy + amount_chromium_second_alloy
def total_weight := 15 + 40

/-- Theorem stating the percentage of chromium -/
theorem chromium_percentage :
  (total_chromium / total_weight) * 100 = 9.09 := by
  sorry

end chromium_percentage_l372_372918


namespace james_money_left_l372_372800

theorem james_money_left 
  (earnings_per_week : ℕ)
  (weeks_saved : ℕ)
  (video_game_fraction : ℝ)
  (book_fraction : ℝ)
  (h1 : earnings_per_week = 10)
  (h2 : weeks_saved = 4)
  (h3 : video_game_fraction = 1/2)
  (h4 : book_fraction = 1/4) : 
  let total_savings := earnings_per_week * weeks_saved in
  let after_video_game := total_savings * video_game_fraction in
  let after_book := after_video_game * book_fraction in
  let remaining_money := after_video_game - after_book in
  remaining_money = 15 :=
by
  sorry

end james_money_left_l372_372800


namespace lloyd_total_hours_worked_l372_372830

-- Conditions
def regular_hours_per_day : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.5
def total_earnings : ℝ := 67.5

-- Proof problem
theorem lloyd_total_hours_worked :
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_earnings := regular_hours_per_day * regular_rate
  let earnings_from_overtime := total_earnings - regular_earnings
  let hours_of_overtime := earnings_from_overtime / overtime_rate
  let total_hours := regular_hours_per_day + hours_of_overtime
  total_hours = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l372_372830


namespace function_monotonically_increasing_implies_a_ge_one_l372_372710

theorem function_monotonically_increasing_implies_a_ge_one 
    (a : ℝ)
    (f : ℝ → ℝ)
    (h_f : ∀ x ∈ Ioo 0 2, (f x = (1 / 3) * a * x^3 - x^2 + x) ∧ 
                            ∀ y ∈ Ioo 0 2, (f' y ≥ 0)) :
    a ≥ 1 :=
sorry

end function_monotonically_increasing_implies_a_ge_one_l372_372710


namespace find_a_l372_372349

-- Define the function f
def f (a x : ℝ) : ℝ :=
  a * x - (3 / 2) * x ^ 2

-- Conditions
def cond1 (a : ℝ) : Prop :=
  ∀ x, f a x ≤ 1 / 6

def cond2 (a : ℝ) : Prop :=
  ∀ x, (1 / 4 ≤ x ∧ x ≤ 1 / 2) → f a x ≥ 1 / 8

-- Statement to prove
theorem find_a (a : ℝ) (h1 : cond1 a) (h2 : cond2 a) : a = 1 := by
  sorry

end find_a_l372_372349


namespace coplanar_lines_condition_l372_372462

theorem coplanar_lines_condition (h : ℝ) : 
  (∃ c : ℝ, 
    (2 : ℝ) = 3 * c ∧ 
    (-1 : ℝ) = c ∧ 
    (h : ℝ) = -2 * c) ↔ 
  (h = 2) :=
by
  sorry

end coplanar_lines_condition_l372_372462


namespace inequality_sum_squares_l372_372559

theorem inequality_sum_squares {n : ℕ} (h1 : n ≥ 2) 
  (a : ℕ → ℝ) (h2 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h3 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i ≤ a j) :
  (∑ i in finset.range n, ∑ j in finset.range (i + 1, a,(a i + a (j + 1))^2 * (1 / (i + 1)^2 + 1 / (j + 1)^2))) ≥ 
  4 * (n - 1) * ∑ i in finset.range n, a (i + 1)^2 / ((i + 1)^2) := sorry

end inequality_sum_squares_l372_372559


namespace coeff_of_reciprocal_cube_in_expansion_l372_372398

theorem coeff_of_reciprocal_cube_in_expansion : 
  (finset.sum (finset.range 10) (λ r : ℕ, if r = 4 then ↑((-1)^r * (nat.choose 9 r)) else 0)) = (126 : ℤ) :=
sorry

end coeff_of_reciprocal_cube_in_expansion_l372_372398


namespace find_x_if_parallel_l372_372727

variables (x : ℝ)

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, x)

theorem find_x_if_parallel : (∃ k : ℝ, a = k • b) → x = -1/2 :=
by
  sorry

end find_x_if_parallel_l372_372727


namespace reasoning_correct_l372_372488

theorem reasoning_correct :
  (∀ (names_not_correct language_not_correct affairs_not_succeed rituals_not_flourish punishments_not_applied people_no_standards: Prop),
    (names_not_correct → language_not_correct) →
    (language_not_correct → affairs_not_succeed) →
    (affairs_not_succeed → rituals_not_flourish) →
    (rituals_not_flourish → punishments_not_applied) →
    (punishments_not_applied → people_no_standards) →
    (names_not_correct → people_no_standards)) →
  ReasoningCommonSense :=
by
  intros names_not_correct language_not_correct affairs_not_succeed rituals_not_flourish punishments_not_applied people_no_standards
  intros h1 h2 h3 h4 h5 h6
  -- Proof would proceed here
  sorry

end reasoning_correct_l372_372488


namespace ellipse_eccentricity_l372_372041

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372041


namespace number_of_polynomial_functions_eq_2_l372_372733

-- Define the problem
def polynomial_of_degree_exactly_3 (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ g = λ x, a * x^3 + b * x^2 + c * x + d

-- Define the functional condition
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x^3) = (g x)^3

-- The main statement that there are exactly 2 such polynomial functions
theorem number_of_polynomial_functions_eq_2 :
  {g : ℝ → ℝ // polynomial_of_degree_exactly_3 g ∧ satisfies_condition g}.card = 2 :=
by sorry

end number_of_polynomial_functions_eq_2_l372_372733


namespace ham_cycle_in_G_l372_372386

variables {V : Type} [fintype V] [decidable_eq V]

structure graph (V : Type) :=
(adj : V → V → Prop)
(symm : symmetric adj . obviously)

def degree (G : graph V) (v : V) : ℕ :=
finset.card {w | G.adj v w}

def G_plane (G : graph V) (n : ℕ) :=
∀ v1 v2 : V, G.adj v1 v2 → degree G v1 + degree G v2 ≥ n

variables (G G' : graph (fin (100)))

theorem ham_cycle_in_G' (H : ∃ p : list (fin 100), p.nodup ∧ p.head = p.nth 99 ∧ p.tail = p.init ∧ (∀ i j : (fin 100), ((p.nth i.succ = p.nth j.succ) ∧ i.succ < 100 ∧ j.succ < 100) → (G'.adj p.nth i p.nth i.succ) ∧ (G'.adj p.nth j p.nth j.succ)))) :
∃ q : list (fin 100), q.nodup ∧ q.head = q.nth 99 ∧ q.tail = q.init ∧ (∀ i j : (fin 100), ((q.nth i.succ = q.nth j.succ) ∧ i.succ < 100 ∧ j.succ < 100) → (G.adj q.nth i q.nth i.succ) ∧ (G.adj q.nth j q.nth j.succ)) :=
sorry

end ham_cycle_in_G_l372_372386


namespace count_odds_between_300_and_600_l372_372732

theorem count_odds_between_300_and_600 : 
  let a := 301
      d := 2
      l := 599 in
  ∃ n, a + (n - 1) * d = l ∧ n = 150 := by
  sorry

end count_odds_between_300_and_600_l372_372732


namespace circle_tangent_prime_radii_l372_372249

theorem circle_tangent_prime_radii :
  let r := [2, 3, 5] in
  ∀ x ∈ r, Nat.Prime x ∧ x < 120 ∧ 120 % x = 0 → r.length = 3 :=
by
  sorry

end circle_tangent_prime_radii_l372_372249


namespace product_of_roots_eq_negative_forty_nine_l372_372653

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372653


namespace length_EF_l372_372312

theorem length_EF
  (AB CD GH EF : ℝ)
  (h1 : AB = 180)
  (h2 : CD = 120)
  (h3 : AB = 2 * GH)
  (h4 : CD = 2 * EF) :
  EF = 45 :=
by
  sorry

end length_EF_l372_372312


namespace revenue_ratio_l372_372184

variable (R_d : ℝ) (R_n : ℝ) (R_j : ℝ)

theorem revenue_ratio
  (nov_cond : R_n = 2 / 5 * R_d)
  (jan_cond : R_j = 1 / 2 * R_n) :
  R_d = 10 / 3 * ((R_n + R_j) / 2) := by
  -- Proof steps go here
  sorry

end revenue_ratio_l372_372184


namespace ellipse_eccentricity_l372_372034

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372034


namespace time_to_cover_length_l372_372555

/-- Define the speeds and length as given in the problem -/
def speed_escalator : ℝ := 11
def length_escalator : ℝ := 140
def speed_person : ℝ := 3

/-- Define the combined speed -/
def combined_speed : ℝ := speed_escalator + speed_person

/-- Define the time taken to cover the length at the combined speed -/
def time_taken : ℝ := length_escalator / combined_speed

/-- Prove that the time taken to cover the entire length is 10 seconds -/
theorem time_to_cover_length : time_taken = 10 :=
by
  -- Proof will be filled here
  sorry

end time_to_cover_length_l372_372555


namespace polygon_has_9_diagonals_has_6_sides_l372_372219

theorem polygon_has_9_diagonals_has_6_sides :
  ∀ (n : ℕ), (∃ D : ℕ, D = n * (n - 3) / 2 ∧ D = 9) → n = 6 := 
by
  sorry

end polygon_has_9_diagonals_has_6_sides_l372_372219


namespace find_a_l372_372054

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372054


namespace pumpkin_pie_price_l372_372392

theorem pumpkin_pie_price :
  ∃ P : ℝ, P = 5 ∧ 
  (let A := 8 in        -- pumpkin pie cut into 8 pieces
   let B := 6 in        -- custard pie cut into 6 pieces
   let C := 6 in        -- price per slice of custard pie
   let D := 4 in        -- number of pumpkin pies sold
   let E := 5 in        -- number of custard pies sold
   let F := 340 in      -- total revenue from sales
   F = (A * D * P + B * E * C))
  := sorry

end pumpkin_pie_price_l372_372392


namespace cardinality_of_set_B_l372_372723

-- Define the set A as described in the condition
def A : Set ℕ := {x | x^2 - 7 * x < 0 ∧ 0 < x ∧ x ∈ ℕ}

-- Define the set B based on set A and condition on y
def B : Set ℕ := {y | (6 % y = 0) ∧ y ∈ A}

-- Assert that the cardinality of set B is 4
theorem cardinality_of_set_B : (B.toFinset.card = 4) := sorry

end cardinality_of_set_B_l372_372723


namespace abs_neg_two_l372_372512

def abs (x : ℝ) : ℝ :=
if x ≥ 0 then x else -x

theorem abs_neg_two : abs (-2) = 2 := 
by
  sorry

end abs_neg_two_l372_372512


namespace find_a_l372_372068

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372068


namespace bob_show_dogs_l372_372243

theorem bob_show_dogs:
  ∃ x : ℕ, 
  let cost := 250 * x in
  let revenue := 6 * 350 in
  let profit := revenue - cost in
  profit = 1600 → x = 2 :=
by
  sorry

end bob_show_dogs_l372_372243


namespace same_curve_option_B_l372_372550

theorem same_curve_option_B : 
  (∀ x y : ℝ, |y| = |x| ↔ y = x ∨ y = -x) ∧ (∀ x y : ℝ, y^2 = x^2 ↔ y = x ∨ y = -x) :=
by
  sorry

end same_curve_option_B_l372_372550


namespace incorrect_statements_l372_372910

theorem incorrect_statements :
  (¬ (∀ x : ℝ, x < 1 → x^2 < 1 ↔ ∃ x : ℝ, x ≥ 1 ∧ x^2 ≥ 1)) ∧
  (¬ ∀ a : ℝ, (∀ x : ℝ, (x = 2 / a ↔ a ≠ 0) → x ∈ ({-2, 1} : set ℝ) ∧ x ∈ {t : ℝ | a * t = 2}) → (a = -1 ∨ a = 2)) ∧
  (0 < a ∧ a < 6 → ∃ α β : ℝ, (3*α^2 + a*(a-6)*α - 3 = 0 ∧ 1 < α ∧ β < 1 ∧ α*β = 0) ↔ (a < 0 ∨ a > 6)) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y - x*y + 8 = 0 → x + 4*y ≥ 17) :=
  sorry

end incorrect_statements_l372_372910


namespace simplify_square_roots_l372_372480

theorem simplify_square_roots : 
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) :=
by
  -- Use the conditions to break down each square root into simpler forms
  have h1 : Real.sqrt 18 = 3 * Real.sqrt 2, 
  { sorry },
  have h2 : Real.sqrt 32 = 4 * Real.sqrt 2, 
  { sorry },
  -- Simplify the given expression
  calc Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2
      = (3 * Real.sqrt 2) - (4 * Real.sqrt 2) + Real.sqrt 2 : by rw [h1, h2]
  ... = (3 - 4 + 1) * Real.sqrt 2 : by ring
  ... = 0 * Real.sqrt 2 : by ring
  ... = 0 : by ring

end simplify_square_roots_l372_372480


namespace find_magnitude_of_b_l372_372107

-- Define the vectors and conditions first
structure Vector2 (α : Type) := (x : α) (y : α)

variables (u v : Vector2 ℝ)
def a : Vector2 ℝ := ⟨2, 0⟩

-- The magnitude of a vector
def magnitude (v : Vector2 ℝ) : ℝ := real.sqrt (v.x ^ 2 + v.y ^ 2)

-- The dot product of two vectors
def dot_product (v w : Vector2 ℝ) : ℝ := v.x * w.x + v.y * w.y

-- Given conditions
def angle_a_b := 60.0
def magnitude_sum_condition := 2 * real.sqrt 3

theorem find_magnitude_of_b :
  ∃ b : Vector2 ℝ, magnitude (a + ⟨2 * b.x, 2 * b.y⟩) = magnitude_sum_condition → magnitude b = 1 :=
sorry

end find_magnitude_of_b_l372_372107


namespace clarence_oranges_l372_372252

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l372_372252


namespace convex_polygon_diagonals_l372_372465

theorem convex_polygon_diagonals (n : ℕ) (h_convex : ConvexPolygon n) 
(h_diag_intersect : ∀ d₁ d₂ ∈ selected_diagonals, ∃ v ∈ vertices, d₁ ∩ d₂ = {v}) : 
∀ d, d ∈ selected_diagonals → d.card ≤ n := 
sorry

end convex_polygon_diagonals_l372_372465


namespace range_of_k_l372_372320

theorem range_of_k (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ ⦃x y⦄, 0 ≤ x → x < y → f y < f x) 
  (h_inequality : ∀ x, f (k * x ^ 2 + 2) + f (k * x + k) ≤ 0) : 0 ≤ k :=
sorry

end range_of_k_l372_372320


namespace Sally_lost_20_Pokemon_cards_l372_372845

theorem Sally_lost_20_Pokemon_cards (original_cards : ℕ) (received_cards : ℕ) (final_cards : ℕ) (lost_cards : ℕ) 
  (h1 : original_cards = 27) 
  (h2 : received_cards = 41) 
  (h3 : final_cards = 48) 
  (h4 : original_cards + received_cards - lost_cards = final_cards) : 
  lost_cards = 20 := 
sorry

end Sally_lost_20_Pokemon_cards_l372_372845


namespace new_average_l372_372458

variable (avg9 : ℝ) (score10 : ℝ) (n : ℕ)
variable (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9)

theorem new_average (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9) :
  ((n * avg9 + score10) / (n + 1)) = 82 :=
by
  rw [h, h10, n9]
  sorry

end new_average_l372_372458


namespace overall_gain_is_1000_l372_372577

def house_selling_price : ℝ := 15000
def store_selling_price : ℝ := 18000
def house_loss_percent : ℝ := 0.25
def store_gain_percent : ℝ := 0.50

def house_cost_price := house_selling_price / (1 - house_loss_percent)
def store_cost_price := store_selling_price / (1 + store_gain_percent)
def total_cost_price := house_cost_price + store_cost_price
def total_selling_price := house_selling_price + store_selling_price
def overall_gain_or_loss := total_selling_price - total_cost_price

theorem overall_gain_is_1000 :
  overall_gain_or_loss = 1000 := by
  -- proof skipped
  sorry

end overall_gain_is_1000_l372_372577


namespace circles_relation_l372_372572

-- Define the variables and conditions
variables {r R1 R2 R3 : ℝ} -- r is the radius of the inscribed circle, R1, R2, R3 are the radii of the other circles

-- State the hypothesis and definition of R1, R2, R3
hypothesis (h1 : ∀ (A B C : ℝ), 
  (R1 = r * (real.tan (real.pi/4 - A/4))^2) ∧ 
  (R2 = r * (real.tan (real.pi/4 - B/4))^2) ∧ 
  (R3 = r * (real.tan (real.pi/4 - C/4))^2) ∧ 
  (A + B + C = real.pi) -- angles sum to π (180 degrees)
)

-- Main theorem statement
theorem circles_relation : 
  ∃ R1 R2 R3, (sqrt (R1 * R2) + sqrt (R2 * R3) + sqrt (R3 * R1) = r) := by
  sorry

end circles_relation_l372_372572


namespace distance_X_X_l372_372530

/-
  Define the vertices of the triangle XYZ
-/
def X : ℝ × ℝ := (2, -4)
def Y : ℝ × ℝ := (-1, 2)
def Z : ℝ × ℝ := (5, 1)

/-
  Define the reflection of point X over the y-axis
-/
def X' : ℝ × ℝ := (-2, -4)

/-
  Prove that the distance between X and X' is 4 units.
-/
theorem distance_X_X' : (Real.sqrt (((-2) - 2) ^ 2 + ((-4) - (-4)) ^ 2)) = 4 := by
  sorry

end distance_X_X_l372_372530


namespace product_of_roots_eq_negative_forty_nine_l372_372654

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372654


namespace measure_angle_CDB_l372_372963

noncomputable def m_angle_CDB : ℝ :=
  let x := 6 in
  x

theorem measure_angle_CDB (triangle_is_equilateral : ∀ {A B C : Type} (e : A × A →  ℝ), true)
(pentagon_is_regular : ∀ {A B C D E : Type} (p : A × A → ℝ), true)
(common_side : ∀ {A B : Type}, true)
(pentagon_interior_angle : 108)
(triangle_interior_angle : 60) :
m_angle_CDB = 6 := 
by
  -- Proof omitted
  sorry

end measure_angle_CDB_l372_372963


namespace seq_100th_term_l372_372881

/-- A sequence where each term is a power of 3 or a sum of distinct powers of 3. -/
noncomputable def seq (n : ℕ) : ℕ :=
  let binary_digits : List ℕ := 
    (Nat.bits n).enum.filter_map (λ p, if p.2 = true then some (3^(p.1)) else none)
  binary_digits.sum

theorem seq_100th_term : seq 100 = 981 :=
  by sorry

end seq_100th_term_l372_372881


namespace embankment_additional_days_l372_372234

noncomputable def rate (workers : ℕ) (days : ℕ) : ℝ :=
  1 / (workers * days)

noncomputable def work_done_in_days (workers : ℕ) (rate : ℝ) (days : ℕ) : ℝ :=
  workers * rate * days

noncomputable def remaining_work : ℝ :=
  1 - (work_done_in_days 100 (rate 100 5) 2)

noncomputable def productivity_after_reduction (workers_left : ℕ) (reduction : ℝ) (rate : ℝ) : ℝ :=
  workers_left * reduction * rate

noncomputable def additional_days_needed (remaining_work : ℝ) (workers_left : ℕ) (reduction : ℝ) (rate : ℝ) : ℕ :=
  (remaining_work / (productivity_after_reduction workers_left reduction rate)).to_nat

theorem embankment_additional_days :
  additional_days_needed remaining_work 60 (3 / 4) (rate 100 5) = 53 := 
sorry

end embankment_additional_days_l372_372234


namespace jared_march_texts_l372_372417

def T (n : ℕ) : ℕ := ((n ^ 2) + 1) * (n.factorial)

theorem jared_march_texts : T 5 = 3120 := by
  -- The details of the proof would go here, but we use sorry to skip it
  sorry

end jared_march_texts_l372_372417


namespace incenters_and_excenters_cyclic_l372_372427

/-- Definitions of the problem conditions -/
variables (A B C D O : Point)
variables (I1 I2 I3 I4 J1 J2 J3 J4 : Point)
variables [convex_quadrilateral A B C D]
variables [intersect AC BD O]
variables [I1 = incenter (triangle A O B), I2 = incenter (triangle B O C)]
variables [I3 = incenter (triangle C O D), I4 = incenter (triangle D O A)]
variables [J1 = excenter (triangle A O B), J2 = excenter (triangle B O C)]
variables [J3 = excenter (triangle C O D), J4 = excenter (triangle D O A)]

/-- Main theorem statement -/
theorem incenters_and_excenters_cyclic :
  (cyclic I1 I2 I3 I4) ↔ (cyclic J1 J2 J3 J4) := by
  sorry

end incenters_and_excenters_cyclic_l372_372427


namespace first_book_cost_correct_l372_372474

noncomputable def cost_of_first_book (x : ℝ) : Prop :=
  let total_cost := x + 6.5
  let given_amount := 20
  let change_received := 8
  total_cost = given_amount - change_received → x = 5.5

theorem first_book_cost_correct : cost_of_first_book 5.5 :=
by
  sorry

end first_book_cost_correct_l372_372474


namespace sum_of_reciprocals_less_than_two_l372_372144

theorem sum_of_reciprocals_less_than_two (n : ℕ) (a : fin n → ℕ)
  (h1 : ∀ i, a i < 1951)
  (h2 : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) > 1951) :
  (∑ i, 1 / (a i : ℝ)) < 2 := 
sorry

end sum_of_reciprocals_less_than_two_l372_372144


namespace sara_grew_4_onions_l372_372090

def onions_grown_by_sally : Nat := 5
def onions_grown_by_fred : Nat := 9
def total_onions_grown : Nat := 18

def onions_grown_by_sara : Nat :=
  total_onions_grown - (onions_grown_by_sally + onions_grown_by_fred)

theorem sara_grew_4_onions :
  onions_grown_by_sara = 4 :=
by
  sorry

end sara_grew_4_onions_l372_372090


namespace ellipse_eccentricity_l372_372035

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372035


namespace roots_of_polynomial_l372_372449

theorem roots_of_polynomial:
  (p q r s : ℝ)
  (h_root : (p, q, r, s).IsRoot x^4 + 10 * x^3 + 20 * x^2 + 15 * x + 6) 
  (h_sum_of_products : p * q + p * r + p * s + q * r + q * s + r * s = 20)
  (h_product_of_roots : p * q * r * s = 6) :
  (1 / (p * q) + 1 / (p * r) + 1/ (p * s) + 1 / (q * r)+ 1 / (q * s) + 1 / (r * s) = 10 / 3) := 
  sorry

end roots_of_polynomial_l372_372449


namespace omicron_radius_l372_372383

theorem omicron_radius 
  (D : ℝ) 
  (hD : D = 120 * 10 ^ (-9)) 
  (convert : ℝ := 10 ^ (-9)) :
  (R : ℝ) = 6 * 10 ^ (-8) :=
by
  sorry

end omicron_radius_l372_372383


namespace find_a_l372_372718

theorem find_a 
  {a : ℝ} 
  (h : ∀ x : ℝ, (ax / (x - 1) < 1) ↔ (x < 1 ∨ x > 3)) : 
  a = 2 / 3 := 
sorry

end find_a_l372_372718


namespace range_a_satisfying_inequality_l372_372935

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f(-x) = -f(x)
axiom decreasing_f : ∀ ⦃x y : ℝ⦄, -1 < x → x < y → y < 1 → f(y) < f(x)

theorem range_a_satisfying_inequality :
  {a : ℝ | f(1-a) + f(1-a^2) < 0} = set.Ioo 0 1 :=
sorry

end range_a_satisfying_inequality_l372_372935


namespace odd_function_property_l372_372369

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then log (2 - x) / log 2 else 0

theorem odd_function_property (h_odd : ∀ x : ℝ, f (-x) = -f (x)) : f 0 + f 2 = -2 :=
by 
  -- Assume the odd function property
  have f_0 : f 0 = 0 := 
    begin
      have h : f (0 : ℝ) = -f 0 := h_odd 0,
      rw [neg_eq_iff_neg_eq, h],
      simp only [neg_zero],
    end,
  have f_2 : f 2 = -2 :=
    by 
      have f_neg2 : f (-2) = log (4) / log 2 :=
        by rewrite neg_neg; simp [f],
      have log2_4 : log 4 / log 2 = 2 := 
        by rewrite [log_div_log_of_base (by linarith)] at f_neg2,
      rw h_odd at f_neg2,
      exact neg_eq_neg_of_eq f_neg2,
  rw [f_0, f_2],
  exact add_zero (-2),

end odd_function_property_l372_372369


namespace symmetrical_point_xOz_plane_l372_372691

theorem symmetrical_point_xOz_plane:
  ∀ (A Q : ℝ × ℝ × ℝ), 
    A = (-1, 2, 3) → 
    (Q.1 = A.1 ∧ Q.3 = A.3 ∧ Q.2 = -A.2) → 
    Q = (-1, -2, 3) :=
by 
  sorry

end symmetrical_point_xOz_plane_l372_372691


namespace proof_problem_l372_372496

noncomputable def exists_abcd (n : ℕ) (x : Fin n → ℝ) : Prop :=
  n > 3 ∧ (∑ i, x i = 0) ∧ (∑ i, (x i)^2 = 1) → ∃ a b c d ∈ (Finset.univ : Finset (Fin n)),
    a + b + c + n * a * b * c ≤ ∑ i, (x i)^3 ∧ ∑ i, (x i)^3 ≤ a + b + d + n * a * b * d

theorem proof_problem (n : ℕ) (x : Fin n → ℝ) (h₁ : n > 3) (h₂ : ∑ i, x i = 0) (h₃ : ∑ i, (x i)^2 = 1) :
  exists_abcd n x :=
begin
  sorry
end

end proof_problem_l372_372496


namespace complex_product_polar_form_l372_372259

theorem complex_product_polar_form :
  (∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 360 ∧
    (4 * complex.cis (real.pi / 4)) * (-5 * complex.cis (72 * real.pi / 180)) = r * complex.cis (θ * real.pi / 180)) :=
sorry

end complex_product_polar_form_l372_372259


namespace max_students_with_extra_credit_l372_372456

theorem max_students_with_extra_credit (n : ℕ) (h_n : n = 150)
(s : ℕ → ℕ) (score_10 : ∀ i, i < 149 → s i = 10)
(score_2 : s 149 = 2) :
  (∑ i in finset.range n, s i) / n < 10 :=
sorry

end max_students_with_extra_credit_l372_372456


namespace abs_z2_minus_2z_eq_2_l372_372753

theorem abs_z2_minus_2z_eq_2 (z : ℂ) (h : z = 1 + complex.I) : abs (z^2 - 2*z) = 2 := by
  sorry

end abs_z2_minus_2z_eq_2_l372_372753


namespace Mn_eq_Mn1_iff_not_power_of_prime_l372_372445

def is_lcm_sequence (M : ℕ → ℕ) :=
  ∀ n, M n = Nat.lcm (List.range (n + 1))

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ p k : ℕ, Nat.prime p ∧ n = p ^ k

theorem Mn_eq_Mn1_iff_not_power_of_prime (M : ℕ → ℕ) (hM : is_lcm_sequence M) (n : ℕ) :
  M (n - 1) = M n ↔ ¬ is_power_of_prime n := sorry

end Mn_eq_Mn1_iff_not_power_of_prime_l372_372445


namespace find_a_l372_372022

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372022


namespace probability_of_sin_ge_cos_l372_372472

theorem probability_of_sin_ge_cos (x : ℝ) (h : x ∈ set.Icc 0 real.pi) :
  (set.Icc (real.pi / 4) real.pi).measure / set.Icc 0 real.pi.measure = 3 / 4 :=
sorry

end probability_of_sin_ge_cos_l372_372472


namespace product_of_solutions_product_of_all_t_l372_372672

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372672


namespace rocket_max_speed_l372_372153

theorem rocket_max_speed (M m : ℝ) (h : 2000 * Real.log (1 + M / m) = 12000) : 
  M / m = Real.exp 6 - 1 := 
by {
  sorry
}

end rocket_max_speed_l372_372153


namespace common_method_for_statistical_analysis_l372_372208

/-- 
We are given the following methods for statistical analysis:
A: Regression analysis
B: Correlation coefficient analysis
C: Residual analysis
D: Correlation index analysis

We need to prove that the common method for statistical analysis of two variables with a correlation is Regression analysis.
-/
theorem common_method_for_statistical_analysis :
  "The common method for statistical analysis of two variables with a correlation is Regression analysis" :=
by
  -- Since we are given that:
  -- Correlation coefficient analysis and Correlation index analysis are used to judge the fitting effect of a model.
  -- Residual analysis is also used to judge the fitting effect of a model.
  -- Hence, the only remaining applicable method is Regression analysis.
  sorry

end common_method_for_statistical_analysis_l372_372208


namespace triangle_area_l372_372414

theorem triangle_area (a b c : ℝ) (A : ℝ) (ha : a = 2) (hb : b = 2 * c) (hcosA : real.cos A = 1 / 4) (h : 0 < A ∧ A < real.pi) :
  let S := (1 / 2) * b * c * real.sin A
  in S = sqrt 15 / 4 :=
by
  sorry

end triangle_area_l372_372414


namespace range_of_a_l372_372308

variable (a x : ℝ)

def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4

def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

theorem range_of_a (h : ∀ (x : ℝ), p a x → q x) : a <= -2 ∨ a >= 7 := 
by sorry

end range_of_a_l372_372308


namespace minimum_cookies_eaten_by_xiao_wang_l372_372391

theorem minimum_cookies_eaten_by_xiao_wang :
  ∃ (x : ℕ), ∀ (n : ℕ),
  let k := (2 / 3 : ℚ) * ((2 / 3 : ℚ) * ((2 / 3 : ℚ) * (x - 1) - 1) - 1),
      evening_blocks := k - 1,
      cookies_each := (evening_blocks / 3)
  in cookies_each = 33 :=
begin
  sorry
end

end minimum_cookies_eaten_by_xiao_wang_l372_372391


namespace remainder_of_expression_l372_372557

theorem remainder_of_expression (k : ℤ) (hk : 0 < k) :
  (4 * k * (2 + 4 + 4 * k) + 3) % 2 = 1 :=
by
  sorry

end remainder_of_expression_l372_372557


namespace calc_expression_correct_l372_372977

noncomputable def calc_expression : Real :=
  Real.sqrt 8 - (1 / 3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2

theorem calc_expression_correct :
  calc_expression = 3 - Real.sqrt 3 :=
sorry

end calc_expression_correct_l372_372977


namespace find_a_l372_372062

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372062


namespace caden_total_euros_l372_372613

-- Definitions of conditions
variables (pennies nickels dimes quarters half_dollars : ℕ)
variables (total_money_dollars total_money_euros : ℝ)

def condition1 := pennies = 160
def condition2 := quarters = 2.5 * dimes
def condition3 := nickels = 7 * dimes
def condition4 := pennies = 4 * nickels
def condition5 := half_dollars = 1.5 * quarters
def exchange_rate := 0.85

def total_dollars := (160 * 0.01) + (40 * 0.05) + (5 * 0.10) + (12 * 0.25) + (18 * 0.50)
def total_euros := total_dollars * exchange_rate

-- Prove the total amount in euros
theorem caden_total_euros : condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ (total_euros = 13.69) := by
  sorry

end caden_total_euros_l372_372613


namespace arithmetic_sequence_k_value_l372_372453

theorem arithmetic_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (S_pos : S 2016 > 0) (S_neg : S 2017 < 0)
  (H : ∀ n, |a n| ≥ |a 1009| ): k = 1009 :=
sorry

end arithmetic_sequence_k_value_l372_372453


namespace product_of_roots_eq_negative_forty_nine_l372_372657

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372657


namespace product_of_values_t_squared_eq_49_l372_372649

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372649


namespace evaporated_water_l372_372950

theorem evaporated_water : 
  ∀ (initial_water drained_by_bob rain_added actual_water evaporated : ℕ),
    initial_water = 6000 ->
    drained_by_bob = 3500 ->
    rain_added = (3 * 350) ->
    actual_water = 1550 ->
    evaporated = initial_water - drained_by_bob + rain_added - actual_water ->
    evaporated = 2000 :=
by
  intros initial_water drained_by_bob rain_added actual_water evaporated
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end evaporated_water_l372_372950


namespace inequality_pos_real_l372_372698

theorem inequality_pos_real (
  a b c : ℝ
) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  abc ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧ 
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end inequality_pos_real_l372_372698


namespace school_A_original_students_l372_372151

theorem school_A_original_students 
  (x y : ℕ) 
  (h1 : x + y = 864) 
  (h2 : x - 32 = y + 80) : 
  x = 488 := 
by 
  sorry

end school_A_original_students_l372_372151


namespace birds_on_the_fence_l372_372860

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l372_372860


namespace part1_part2_l372_372717

def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

theorem part1 (a b c x : ℝ) (h1 : |a - b| > c) : f x a b > c :=
  by sorry

theorem part2 (a : ℝ) (h1 : ∃ (x : ℝ), f x a 1 < 2 - |a - 2|) : 1/2 < a ∧ a < 5/2 :=
  by sorry

end part1_part2_l372_372717


namespace not_general_prob1_not_general_prob2_l372_372471

variables {Ω : Type} {P : MeasureTheory.Measure Ω} 
variables {A B : Set Ω}

open MeasureTheory
open ProbabilityTheory

-- Mathematical translations of the problem
theorem not_general_prob1 (A : Set Ω) : 
  ∃ B : Set Ω, P[B|A] + P[B|Aᶜ] ≠ 1 :=
begin
  use ∅,
  simp [condProbability_simp],
  exact zero_ne_one,
end

theorem not_general_prob2 (A : Set Ω) :
  ∃ B : Set Ω, P[B|A] + P[Bᶜ|Aᶜ] ≠ 1 :=
begin
  use A,
  simp [condProbability_simp],
  exact two_ne_one,
end

end not_general_prob1_not_general_prob2_l372_372471


namespace find_a_l372_372019

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372019


namespace sum_ceil_sqrt_l372_372278

def ceil_sqrt (x : ℝ) : ℕ := ⌈Real.sqrt x⌉₊ -- defining a ceiling of square root function

theorem sum_ceil_sqrt :
  (∑ n in Finset.range (51 - 5) + 5, ceil_sqrt n) = 252 :=
by {
  sorry
}

end sum_ceil_sqrt_l372_372278


namespace arithmetic_mean_of_multiples_of_5_l372_372160

-- Define the sequence and its properties
def a₁ : ℕ := 10
def d : ℕ := 5
def aₙ : ℕ := 95

-- Find number of terms in the sequence
def n: ℕ := (aₙ - a₁) / d + 1

-- Define the sum of the sequence
def S := n * (a₁ + aₙ) / 2

-- Define the arithmetic mean
def arithmetic_mean := S / n

-- Prove the arithmetic mean
theorem arithmetic_mean_of_multiples_of_5 : arithmetic_mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_multiples_of_5_l372_372160


namespace tetrahedron_volume_l372_372491

noncomputable def volume_of_tetrahedron (S1 S2 a α : ℝ) : ℝ :=
  (2 * S1 * S2 * Real.sin α) / (3 * a)

theorem tetrahedron_volume (S1 S2 a α : ℝ) :
  a > 0 → S1 > 0 → S2 > 0 → α ≥ 0 → α ≤ Real.pi → volume_of_tetrahedron S1 S2 a α =
  (2 * S1 * S2 * Real.sin α) / (3 * a) := 
by
  intros
  -- The proof is omitted here.
  sorry

end tetrahedron_volume_l372_372491


namespace birth_death_rate_interval_l372_372389

theorem birth_death_rate_interval
  (b_rate : ℕ) (d_rate : ℕ) (population_increase_one_day : ℕ) (seconds_in_one_day : ℕ)
  (net_increase_per_t_seconds : ℕ) (t : ℕ)
  (h1 : b_rate = 5)
  (h2 : d_rate = 3)
  (h3 : population_increase_one_day = 86400)
  (h4 : seconds_in_one_day = 86400)
  (h5 : net_increase_per_t_seconds = b_rate - d_rate)
  (h6 : population_increase_one_day = net_increase_per_t_seconds * (seconds_in_one_day / t)) :
  t = 2 :=
by
  sorry

end birth_death_rate_interval_l372_372389


namespace parkway_girls_not_playing_soccer_l372_372187

theorem parkway_girls_not_playing_soccer :
  ∀ (total_students boys total_soccer_players boys_playing_soccer : ℕ)
    (boy_soccer_percent : ℝ),
  total_students = 470 →
  boys = 300 →
  total_soccer_players = 250 →
  boy_soccer_percent = 0.86 →
  boys_playing_soccer = (boy_soccer_percent * total_soccer_players).to_nat →
  (total_students - boys) - (total_soccer_players - boys_playing_soccer) = 135 :=
sorry

end parkway_girls_not_playing_soccer_l372_372187


namespace part_a_part_b_l372_372921

-- Definitions representing the data structure and conditions
variables {ι : Type*} {n : ℕ} (S : Set (Fin n → ℝ) → ℝ)
variable (I : ∀ (k : Fin n), Set (Fin n → ℝ))
variable (M : Fin n → ℝ)

-- Given conditions
variable (f_intersection_area : ∀ {k : Fin n} (I : Fin k → ℝ), ℝ)
variables {k m : ℕ}

-- Definitions
definition area_covered (n : ℕ) (S : Set (Fin n → ℝ) → ℝ) : ℝ :=
  ∑ i in Finset.range n, S (I i)

-- Part (a) statement
theorem part_a (h_mk_eq_sum : ∀ k, M k = ∑ i in Finset.range n, S (I i)) :
  S (I 0) = ∑ k in Finset.range(n + 1), (-1)^k * M k :=
sorry

-- Part (b) statement
theorem part_b (h_mk_eq_sum : ∀ k, M k = ∑ i in Finset.range n, S (I i)) (m : ℕ) :
  (even m → S (I 0) ≥  ∑ k in Finset.range(m + 1), (-1)^k * M k) ∧
  (odd m → S (I 0) ≤ ∑ k in Finset.range(m + 1), (-1)^k * M k) :=
sorry

end part_a_part_b_l372_372921


namespace find_a_and_x_l372_372944

theorem find_a_and_x (x a : ℕ) (h₁ : x > 0) (h₂ : sqrt x = 3 * a - 14) (h₃ : sqrt x = a - 2) : a = 4 ∧ x = 4 := by
  sorry

end find_a_and_x_l372_372944


namespace geometric_progression_sixth_term_proof_l372_372498

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l372_372498


namespace no_integer_solution_l372_372432

theorem no_integer_solution (n : ℕ) (p : ℕ) (hp_prime : prime p) (hp_gt : p > n + 1) :
  ¬∃ x : ℤ, (1 + (x / (n + 1)) + (x^2 / (2 * n + 1)) + (x^3 / (3 * n + 1)) + ... + (x^p / (p * n + 1)) = 0) := 
sorry

end no_integer_solution_l372_372432


namespace no_three_300_degree_arcs_no_common_points_l372_372839

theorem no_three_300_degree_arcs_no_common_points :
  ∀ (s : Sphere) (arcs : list (GreatCircleArc s)) (h : ∀ a ∈ arcs, a.measure = 300)
     (np : ∀ (a1 a2 : GreatCircleArc s), a1 ∈ arcs → a2 ∈ arcs → a1 ≠ a2 → a1 ∩ a2 = ∅),
  arcs.length ≠ 3 := 
sorry

end no_three_300_degree_arcs_no_common_points_l372_372839


namespace smallest_number_of_eggs_proof_l372_372176

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l372_372176


namespace find_c_find_cos_2B_minus_pi_over_4_l372_372696

variable (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (a b c : Real) (A : Real) : Prop :=
  a = 4 * Real.sqrt 3 ∧
  b = 6 ∧
  Real.cos A = -1 / 3

-- Proof of question 1
theorem find_c (h : conditions a b c A) : c = 2 :=
sorry

-- Proof of question 2
theorem find_cos_2B_minus_pi_over_4 (h : conditions a b c A) (B : Real) :
  (angle_opp_b : b = Real.sin B) → -- This is to ensure B is the angle opposite to side b
  Real.cos (2 * B - Real.pi / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end find_c_find_cos_2B_minus_pi_over_4_l372_372696


namespace minimum_moves_triangle_l372_372146

def convex_polygon (n : ℕ) : Prop := 3 * n > 0

def robot (n : ℕ) :=
  { i : ℕ // 1 ≤ i ∧ i ≤ 3 * n }

def laser_beam (n : ℕ) (r : robot n) : robot n → Prop := 
  λ target, true -- A placeholder, meaning any target is valid

def is_triangle (n : ℕ) (A B C : robot n) : Prop :=
  laser_beam n A B ∧ laser_beam n B C ∧ laser_beam n C A

def min_moves_for_triangles (n : ℕ) : ℕ := (9 * n^2 - 7 * n) / 2

theorem minimum_moves_triangle (n : ℕ) :
  convex_polygon n → (∀ i, i ∈ finset.range n → ∃ A B C, is_triangle n A B C) →
  ∃ moves, moves = min_moves_for_triangles n :=
by
  intros
  sorry

end minimum_moves_triangle_l372_372146


namespace smallest_b_for_perfect_square_l372_372164

theorem smallest_b_for_perfect_square : ∃ (b : ℕ), b > 4 ∧ (∃ k, (2 * b + 4) = k * k) ∧
                                             ∀ (b' : ℕ), b' > 4 ∧ (∃ k, (2 * b' + 4) = k * k) → b ≤ b' :=
by
  sorry

end smallest_b_for_perfect_square_l372_372164


namespace n_not_prime_find_n_value_l372_372943

-- Define conditions 
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Part (a)
theorem n_not_prime (n : ℕ) (h₁ : 0 < n) (h₂ : is_perfect_square (n * (n + 2013))) : ¬ prime n :=
sorry

-- Part (b)
theorem find_n_value : ∃ n : ℕ, is_perfect_square (n * (n + 2013)) ∧ n = 1006 * 1006 :=
⟨1006 * 1006, by {
  use 1006 * 2013 + 1006 * 1006,
  sorry
}⟩

end n_not_prime_find_n_value_l372_372943


namespace geometric_seq_general_term_and_bn_sum_lt_1_l372_372486

theorem geometric_seq_general_term_and_bn_sum_lt_1
  (a : ℕ → ℕ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_geom : ∀ n, a n = 2 ^ n)
  (h_rel : ∀ n, a n * b n = 2 ^ n / (n^2 + n))
  (h_sum : ∀ n, T n = ∑ i in Finset.range (n + 1), b i) :
  ∀ n, T n < 1 :=
by
  sorry

end geometric_seq_general_term_and_bn_sum_lt_1_l372_372486


namespace Sedrach_apple_pies_l372_372476

theorem Sedrach_apple_pies (samples_needed : ℕ) (samples_per_pie : ℕ) (total_people : ℕ) (total_people = samples_needed)
  (samples_per_pie = 10) (samples_needed = 130) : 
  (samples_needed / samples_per_pie) = 13 := 
by sorry

end Sedrach_apple_pies_l372_372476


namespace area_rectangle_l372_372841

-- Given data
variables {DA : ℝ} (GD DH : ℝ) (r : ℝ)
def GH : ℝ := GD + DA + DH

-- Assumptions
axiom DA_val : DA = 20
axiom length_GD_DH : GD = 5 ∧ DH = 5
axiom GH_val : GH = 30
axiom radius_val : r = GH / 2

-- The theorem: the area of rectangle ABCD
theorem area_rectangle (h1 : DA = 20) (h2 : GD = 5) (h3 : DH = 5) : 
  let CD := sqrt (5 * GH):ℝ in
  DA * CD = 100 * sqrt 6 := 
by 
  sorry

end area_rectangle_l372_372841


namespace jinhee_pages_per_day_l372_372418

noncomputable def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  (total_pages + days - 1) / days

theorem jinhee_pages_per_day : 
  ∀ (total_pages : ℕ) (days : ℕ), total_pages = 220 → days = 7 → pages_per_day total_pages days = 32 :=
by 
  intros total_pages days hp hd
  rw [hp, hd]
  -- the computation of the function
  show pages_per_day 220 7 = 32
  sorry

end jinhee_pages_per_day_l372_372418


namespace extremum_at_one_monotonic_increasing_l372_372697

noncomputable def f (x : ℝ) (b : ℝ) := 2 * x + b / x + Real.log x
noncomputable def g (x : ℝ) (a : ℝ) := f x 3 - (3 + a) / x

theorem extremum_at_one (b : ℝ) (h : ∀ x : ℝ, has_deriv_at (f x b) (2 - b / x^2 + 1 / x) x) :
  deriv (f 1 b) = 0 → b = 3 := by
  intro h1
  sorry

theorem monotonic_increasing (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → x ≤ 2 → deriv (g x a) ≥ 0) → a ≥ -3 := by
  intro h2
  sorry

end extremum_at_one_monotonic_increasing_l372_372697


namespace trapezoid_area_l372_372116

-- Definitions based on the given conditions
variable (BD AC h : ℝ)
variable (BD_perpendicular_AC : BD * AC = 0)
variable (BD_val : BD = 13)
variable (h_val : h = 12)

-- Statement of the theorem to prove the area of the trapezoid
theorem trapezoid_area (BD AC h : ℝ)
  (BD_perpendicular_AC : BD * AC = 0)
  (BD_val : BD = 13)
  (h_val : h = 12) :
  0.5 * 13 * 12 = 1014 / 5 := sorry

end trapezoid_area_l372_372116


namespace total_wicks_l372_372958

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l372_372958


namespace total_wicks_20_l372_372960

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l372_372960


namespace log_absinequality_l372_372689

theorem log_absinequality (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 1) (ha : 0 < a) (h_ne_one : a ≠ 1) :
  abs (Real.logBase a (1-x)) > abs (Real.logBase a (1+x)) :=
sorry

end log_absinequality_l372_372689


namespace product_of_values_t_squared_eq_49_l372_372651

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372651


namespace percent_workforce_from_A_l372_372255

variables {A B : ℕ} -- A and B represent the total number of employees in Company A and Company B respectively

-- Conditions
def managers_A : ℕ := 0.10 * A
def managers_B : ℕ := 0.30 * B
def total_managers : ℕ := 0.25 * (A + B)

-- Theorem statement
theorem percent_workforce_from_A (h1 : managers_A + managers_B = total_managers) : 
  (A : ℚ) / (A + B) = 1 / 4 := 
sorry

end percent_workforce_from_A_l372_372255


namespace abs_z_squared_minus_two_z_eq_two_l372_372746

theorem abs_z_squared_minus_two_z_eq_two (z : ℂ) (hz : z = 1 + 1*Complex.i) : 
  |z^2 - 2*z| = 2 :=
begin
  sorry
end

end abs_z_squared_minus_two_z_eq_two_l372_372746


namespace number_with_29_proper_divisors_is_720_l372_372619

theorem number_with_29_proper_divisors_is_720
  (n : ℕ) (h1 : n < 1000)
  (h2 : ∀ d, 1 < d ∧ d < n -> ∃ m, n = d * m):
  n = 720 := by
  sorry

end number_with_29_proper_divisors_is_720_l372_372619


namespace product_of_solutions_of_t_squared_eq_49_l372_372669

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372669


namespace length_of_paper_l372_372938

theorem length_of_paper (width : ℝ) (wraps : ℕ) (initial_diameter : ℝ) (final_diameter : ℝ) : 
  width = 5 ∧ wraps = 600 ∧ initial_diameter = 2 ∧ final_diameter = 10 →
  (∑ i in Finset.range(wraps), (initial_diameter + 2 * i * width / wraps)) * π / 100 = 36 * π :=
by
  sorry

end length_of_paper_l372_372938


namespace racers_in_final_segment_l372_372245

theorem racers_in_final_segment : 
  (initial_racers eliminated_first_segment remaining_after_first_segment 
    eliminated_second_segment remaining_after_second_segment 
    eliminated_third_segment remaining_after_third_segment : ℕ)
  (h1 : initial_racers = 100)
  (h2 : eliminated_first_segment = 10)
  (h3 : remaining_after_first_segment = initial_racers - eliminated_first_segment)
  (h4 : eliminated_second_segment = remaining_after_first_segment / 3)
  (h5 : remaining_after_second_segment = remaining_after_first_segment - eliminated_second_segment)
  (h6 : eliminated_third_segment = remaining_after_second_segment / 2)
  (h7 : remaining_after_third_segment = remaining_after_second_segment - eliminated_third_segment)
  : remaining_after_third_segment = 30 :=
by
  have h8 : remaining_after_first_segment = 90, {
    rw [h1, h2, h3], simp
  },
  have h9 : eliminated_second_segment = 30, {
    rw [h8, h4], norm_num
  },
  have h10 : remaining_after_second_segment = 60, {
    rw [h8, h9, h5], norm_num
  },
  have h11 : eliminated_third_segment = 30, {
    rw [h10, h6], norm_num
  },
  have : remaining_after_third_segment = 30, {
    rw [h10, h11, h7], norm_num
  },
  assumption

end racers_in_final_segment_l372_372245


namespace point_coordinates_correct_l372_372396

def point_coordinates : (ℕ × ℕ) :=
(11, 9)

theorem point_coordinates_correct :
  point_coordinates = (11, 9) :=
by
  sorry

end point_coordinates_correct_l372_372396


namespace radius_of_spheres_in_cone_l372_372795

-- Given Definitions
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15
def tangent_spheres (r : ℝ) : Prop :=
  r = (12 * Real.sqrt 29) / 29

-- Problem Statement
theorem radius_of_spheres_in_cone :
  ∃ r : ℝ, tangent_spheres r :=
sorry

end radius_of_spheres_in_cone_l372_372795


namespace smallest_number_of_eggs_l372_372173

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372173


namespace nine_point_circle_l372_372469

theorem nine_point_circle (A B C H M_A M_B M_C D E F P_A P_B P_C : Type*)
  [IsTriangle A B C] [IsOrthocenter H A B C]
  [Midpoint M_A B C] [Midpoint M_B A C] [Midpoint M_C A B]
  [FootOfAltitude D A B C] [FootOfAltitude E B A C] [FootOfAltitude F C A B]
  [Midpoint P_A A H] [Midpoint P_B B H] [Midpoint P_C C H] :
  Concyclic M_A M_B M_C D E F P_A P_B P_C := sorry

end nine_point_circle_l372_372469


namespace number_of_members_l372_372554

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l372_372554


namespace lifespan_of_bat_l372_372119

theorem lifespan_of_bat (B : ℕ) (h₁ : ∀ B, B - 6 < B)
    (h₂ : ∀ B, 4 * (B - 6) < 4 * B)
    (h₃ : B + (B - 6) + 4 * (B - 6) = 30) :
    B = 10 := by
  sorry

end lifespan_of_bat_l372_372119


namespace prove_ab_ge_5_l372_372430

theorem prove_ab_ge_5 (a b c : ℕ) (h : ∀ x, x * (a * x) = b * x + c → 0 ≤ x ∧ x ≤ 1) : 5 ≤ a ∧ 5 ≤ b := 
sorry

end prove_ab_ge_5_l372_372430


namespace sum_of_powers_of_i_l372_372194

-- Define the imaginary unit and its property
def i : ℂ := Complex.I -- ℂ represents the complex numbers, Complex.I is the imaginary unit

-- The statement we need to prove
theorem sum_of_powers_of_i : i + i^2 + i^3 + i^4 = 0 := 
by {
  -- Lean requires the proof, but we will use sorry to skip it.
  -- Define the properties of i directly or use in-built properties
  sorry
}

end sum_of_powers_of_i_l372_372194


namespace necessary_and_sufficient_condition_l372_372311

noncomputable def A := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 4)^2 ≤ 1}
noncomputable def B (a : ℝ) := {p : ℝ × ℝ | |p.1 - 3| + 2 * |p.2 + 4| ≤ a}

theorem necessary_and_sufficient_condition (a : ℝ) (h : a > 0) :
  A ⊆ B a ↔ a ≥ real.sqrt 5 :=
sorry

end necessary_and_sufficient_condition_l372_372311


namespace prob_sum_greater_than_9_two_dice_l372_372531

theorem prob_sum_greater_than_9_two_dice :
  let outcomes := {(d1, d2) | d1 ∈ finset.range 6 ∧ d2 ∈ finset.range 6},
      favorable := {(d1, d2) ∈ outcomes | d1 + d2 + 2 > 9} in
  (finset.card favorable) / (finset.card outcomes) = 1 / 6 :=
by
  let outcomes := finset.image (λ (d1, d2), (d1 + 1, d2 + 1))
                                (finset.product (finset.range 6) (finset.range 6))
  let favorable := finset.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd > 9) outcomes
  have h_outcomes : outcomes.card = 36 := sorry
  have h_favorable : favorable.card = 6 := sorry
  calc
    (favorable.card : ℝ) / outcomes.card = 6 / 36 : by rw [h_outcomes, h_favorable]
    ... = 1 / 6 : by norm_num

end prob_sum_greater_than_9_two_dice_l372_372531


namespace racers_in_final_segment_l372_372246

theorem racers_in_final_segment : 
  (initial_racers eliminated_first_segment remaining_after_first_segment 
    eliminated_second_segment remaining_after_second_segment 
    eliminated_third_segment remaining_after_third_segment : ℕ)
  (h1 : initial_racers = 100)
  (h2 : eliminated_first_segment = 10)
  (h3 : remaining_after_first_segment = initial_racers - eliminated_first_segment)
  (h4 : eliminated_second_segment = remaining_after_first_segment / 3)
  (h5 : remaining_after_second_segment = remaining_after_first_segment - eliminated_second_segment)
  (h6 : eliminated_third_segment = remaining_after_second_segment / 2)
  (h7 : remaining_after_third_segment = remaining_after_second_segment - eliminated_third_segment)
  : remaining_after_third_segment = 30 :=
by
  have h8 : remaining_after_first_segment = 90, {
    rw [h1, h2, h3], simp
  },
  have h9 : eliminated_second_segment = 30, {
    rw [h8, h4], norm_num
  },
  have h10 : remaining_after_second_segment = 60, {
    rw [h8, h9, h5], norm_num
  },
  have h11 : eliminated_third_segment = 30, {
    rw [h10, h6], norm_num
  },
  have : remaining_after_third_segment = 30, {
    rw [h10, h11, h7], norm_num
  },
  assumption

end racers_in_final_segment_l372_372246


namespace num_satisfying_permutations_l372_372293

open Finset

theorem num_satisfying_permutations : 
  ∃ (s : FinSet (Fin 6 → ℕ)) 
  (h_permutations : ∀ x : s, ∀ k : ℕ, k ∈ FinSet.univ → (x k + k) / 2 < 8!),
  s.card = nat.factorial 6 :=
by
  sorry

end num_satisfying_permutations_l372_372293


namespace correct_statements_about_f_l372_372706

noncomputable def f (x : ℝ) := Real.sin (1/x)

theorem correct_statements_about_f :
  (∀ m ∈ Set.Icc (-1 : ℝ) 1, ∃ᶠ x in Filter.atTop, f x = m) ∧
  (∀ x ∈ Set.Icc (2/Real.pi : ℝ) ∞, MonotoneDecreasing f) ∧
  Set.range f = Set.Icc (-1 : ℝ) 1 :=
by
  sorry

end correct_statements_about_f_l372_372706


namespace proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l372_372484

structure CubeSymmetry where
  planes : Nat
  axes : Nat
  has_center : Bool

def general_cube_symmetry : CubeSymmetry :=
  { planes := 9, axes := 9, has_center := true }

def case_a : CubeSymmetry :=
  { planes := 4, axes := 1, has_center := false }

def case_b1 : CubeSymmetry :=
  { planes := 5, axes := 3, has_center := true }

def case_b2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

def case_c1 : CubeSymmetry :=
  { planes := 3, axes := 0, has_center := false }

def case_c2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

theorem proof_case_a : case_a = { planes := 4, axes := 1, has_center := false } := by
  sorry

theorem proof_case_b1 : case_b1 = { planes := 5, axes := 3, has_center := true } := by
  sorry

theorem proof_case_b2 : case_b2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

theorem proof_case_c1 : case_c1 = { planes := 3, axes := 0, has_center := false } := by
  sorry

theorem proof_case_c2 : case_c2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

end proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l372_372484


namespace find_g_l372_372080

theorem find_g (g : ℕ) (h : g > 0) :
  (1 / 3) = ((4 + g * (g - 1)) / ((g + 4) * (g + 3))) → g = 5 :=
by
  intro h_eq
  sorry 

end find_g_l372_372080


namespace repaired_shoes_lifespan_l372_372569

-- Definitions of given conditions
def cost_repair : Float := 11.50
def cost_new : Float := 28.00
def lifespan_new : Float := 2.0
def percentage_increase : Float := 21.73913043478261 / 100

-- Cost per year of new shoes
def cost_per_year_new : Float := cost_new / lifespan_new

-- Cost per year of repaired shoes
def cost_per_year_repair (T : Float) : Float := cost_repair / T

-- Theorem statement (goal)
theorem repaired_shoes_lifespan (T : Float) (h : cost_per_year_new = cost_per_year_repair T * (1 + percentage_increase)) : T = 0.6745 :=
by
  sorry

end repaired_shoes_lifespan_l372_372569


namespace triangle_equilateral_l372_372773

theorem triangle_equilateral (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
  (angle_B : angle A B C = 60) (condition : b*b = a*c) : 
  is_equilateral_triangle A B C :=
sorry

end triangle_equilateral_l372_372773


namespace price_increase_for_weekly_profit_price_increase_to_maximize_profit_l372_372570

-- Given conditions
def currentPrice := 60
def currentVolume := 300
def decreaseRate := 10
def costPrice := 40

-- Problem 1: Prove the selling price increase for a weekly profit of 6000 yuan
theorem price_increase_for_weekly_profit 
  (x : ℕ) (h : (currentPrice + x - costPrice) * (currentVolume - decreaseRate * x) = 6000) : 
  x = 10 :=
by 
  sorry

-- Problem 2: Prove the selling price increase to maximize profit
theorem price_increase_to_maximize_profit 
  (x : ℕ) (profit : x → ℕ := λ x, (x + currentPrice - costPrice) * (currentVolume - decreaseRate * x)) : 
  profit 5 = (List.max (List.map profit (List.range 11))) := 
by 
  sorry

end price_increase_for_weekly_profit_price_increase_to_maximize_profit_l372_372570


namespace angle_between_a_and_a_plus_2b_l372_372763

open Real

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem angle_between_a_and_a_plus_2b
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 1)
  (hab_ang : angle_between_vectors a b = π / 3) :
  angle_between_vectors a (a + 2 • b) = π / 6 :=
sorry

end angle_between_a_and_a_plus_2b_l372_372763


namespace estate_area_l372_372964

theorem estate_area (scale : ℝ) (len_inch : ℝ) (wid_inch : ℝ) (len_mile corr) (wid_mile corr):
  scale = 300 ∧ len_inch = 6 ∧ wid_inch = 4 → (scale * len_inch * scale * wid_inch = 2160000) := by
  sorry

end estate_area_l372_372964


namespace ellipse_eccentricity_a_l372_372015

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372015


namespace sum_smallest_and_largest_prime_between_1_and_50_l372_372423

noncomputable def smallest_prime_between_1_and_50 : ℕ := 2
noncomputable def largest_prime_between_1_and_50 : ℕ := 47

theorem sum_smallest_and_largest_prime_between_1_and_50 : 
  smallest_prime_between_1_and_50 + largest_prime_between_1_and_50 = 49 := 
by
  sorry

end sum_smallest_and_largest_prime_between_1_and_50_l372_372423


namespace smallest_number_of_eggs_l372_372179

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372179


namespace find_m_for_unique_solution_l372_372634

theorem find_m_for_unique_solution :
  ∃ m : ℝ, (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) ∧ 
  ∀ x : ℝ, (mx - 2 ≠ 0 → (x + 3) / (mx - 2) = x + 1 ↔ ∃! x : ℝ, (mx - 2) * (x + 1) = (x + 3)) :=
sorry

end find_m_for_unique_solution_l372_372634


namespace largest_two_digit_prime_factor_l372_372907

open Nat

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem largest_two_digit_prime_factor :
  let n := binom 210 105 
  ∃ p : ℕ, 10 ≤ p ∧ p < 100 ∧ Prime p ∧ (∃ k, n = k * p) ∧ 
    ∀ q : ℕ, 10 ≤ q ∧ q < 100 ∧ Prime q ∧ (∃ l, n = l * q) -> q ≤ p :=
  sorry

end largest_two_digit_prime_factor_l372_372907


namespace proof_inequality_l372_372808

variable {m n : ℕ}
variable {x : Fin m → ℝ}
variable {y : Fin n → ℝ}

def X : ℝ := (Finset.univ.sum (λ i, x i))
def Y : ℝ := (Finset.univ.sum (λ j, y j))
def A : ℝ := (Finset.univ.sum (λ i, Finset.univ.sum (λ j, |x i - y j|)))
def B : ℝ := (Finset.univ.sum (λ j, Finset.univ.sum (λ l, |y j - y l|)))
def C : ℝ := (Finset.univ.sum (λ i, Finset.univ.sum (λ k, |x i - x k|)))

theorem proof_inequality :
  (2 * X * Y * A) ≥ (X^2 * B + Y^2 * C) :=
sorry

end proof_inequality_l372_372808


namespace polynomial_remainder_l372_372908

def polynomial := λ x : ℝ, 4 * x^7 - 2 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 15
def divisor := λ x : ℝ, 3 * (x - 3)

theorem polynomial_remainder : polynomial 3 = 5457 := 
by
  sorry

end polynomial_remainder_l372_372908


namespace multiples_of_4_l372_372139

theorem multiples_of_4 (n : ℕ) (h : n + 23 * 4 = 112) : n = 20 :=
by
  sorry

end multiples_of_4_l372_372139


namespace sum_of_products_neg_l372_372198

theorem sum_of_products_neg {a : Fin 2011 → ℤ} 
  (h : ∀ i : Fin 2011, a i + ∏ j : Fin 2011, if j = i then 1 else a j < 0) :
  ∀ (s1 s2 : Finset (Fin 2011)), s1 ∪ s2 = Finset.univ → s1 ∩ s2 = ∅ → 
    ((∏ i in s1, a i) + (∏ i in s2, a i) < 0) := 
by {
  sorry
}

end sum_of_products_neg_l372_372198


namespace inequality_of_positive_numbers_l372_372823

variable {a b : ℝ}

theorem inequality_of_positive_numbers (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ sqrt(ab) + sqrt((a^2 + b^2) / 2) :=
sorry

end inequality_of_positive_numbers_l372_372823


namespace div_polynomial_not_div_l372_372681

theorem div_polynomial_not_div (n : ℕ) : ¬ (n + 2) ∣ (n^3 - 2 * n^2 - 5 * n + 7) := by
  sorry

end div_polynomial_not_div_l372_372681


namespace odd_power_divisible_by_sum_l372_372904

theorem odd_power_divisible_by_sum (x y : ℝ) (k : ℕ) (h : k > 0) :
  (x^((2*k - 1)) + y^((2*k - 1))) ∣ (x^(2*k + 1) + y^(2*k + 1)) :=
sorry

end odd_power_divisible_by_sum_l372_372904


namespace final_racers_count_l372_372247

theorem final_racers_count : 
  ∀ (initial_racers eliminated_first third_remaining half_remaining : ℕ),
    initial_racers = 100 →
    eliminated_first = 10 →
    third_remaining = (initial_racers - eliminated_first) / 3 →
    half_remaining = (initial_racers - eliminated_first - third_remaining) / 2 →
  initial_racers - eliminated_first - third_remaining - half_remaining = 30 :=
begin
  intros initial_racers eliminated_first third_remaining half_remaining,
  assume h1 : initial_racers = 100,
  assume h2 : eliminated_first = 10,
  assume h3 : third_remaining = (initial_racers - eliminated_first) / 3,
  assume h4 : half_remaining = (initial_racers - eliminated_first - third_remaining) / 2,
  sorry
end

end final_racers_count_l372_372247


namespace subsets_contain_a_l372_372514

theorem subsets_contain_a (a b : Type) (S : set (Type)) (hS : S = {a, b}) :
  (set.filter (λ (x : set Type), a ∈ x) (set.powerset S)).card = 2 :=
by
  sorry

end subsets_contain_a_l372_372514


namespace ellipse_eccentricity_l372_372040

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372040


namespace values_of_n_for_c_n_zero_l372_372627

noncomputable def sequence_c (a : Fin 8 → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.univ, (a i) ^ n

theorem values_of_n_for_c_n_zero (a : Fin 8 → ℝ)
  (h : ∃∞ n, sequence_c a n = 0) : 
  ∀ n, sequence_c a n = 0 ↔ Odd n := by
  sorry

end values_of_n_for_c_n_zero_l372_372627


namespace driving_routes_l372_372968

-- Define the total number of possible routes given the conditions.
theorem driving_routes (start_points : ℕ) (dest_points : ℕ) (no_turn_back : Prop) 
  (h1 : start_points = 4) (h2 : dest_points = 3) : 
  (start_points * dest_points = 12) :=
by
  rw [h1, h2]
  norm_num

end driving_routes_l372_372968


namespace intersect_at_one_point_l372_372241

-- Define the quadratic and linear functions
def quadratic (b x : ℝ) : ℝ := b * x^2 + b * x + 2
def linear (x : ℝ) : ℝ := 2 * x + 4

-- Define the discriminant of the quadratic equation resulting from setting the polynomials equal
def discriminant (b : ℝ) : ℝ := (b - 2)^2 - 4 * b * (-2)

-- The main theorem we want to prove
theorem intersect_at_one_point (b : ℝ) : discriminant b = 0 ↔ b = -2 := by
  unfold discriminant
  sorry

end intersect_at_one_point_l372_372241


namespace problem1_problem2_l372_372317

-- Define the points A, B, and C
def point_A : ℝ×ℝ := (-3, 0)
def point_B : ℝ×ℝ := (2, 1)
def point_C : ℝ×ℝ := (-2, 3)

-- Define the midpoint formula
def midpoint (p1 p2 : ℝ×ℝ) : ℝ×ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the equation of a line given two points
def line_eq (p1 p2 : ℝ×ℝ) : ℝ → ℝ → Prop :=
  λ x y, (y - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (x - p1.1)

-- Problem statements
theorem problem1 : line_eq point_B point_C = (λ x y, x + 2 * y - 4 = 0) :=
sorry

theorem problem2 :
  let point_D := midpoint point_B point_C
  in line_eq point_A point_D = (λ x y, 2 * x - 3 * y + 6 = 0) :=
sorry

end problem1_problem2_l372_372317


namespace ellipse_eccentricity_l372_372043

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372043


namespace log_sum_geometric_sequence_l372_372816

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem log_sum_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 4 * a 7 + a 5 * a 6 = 18) :
  ∑ i in finset.range 10, Real.logBase 3 (a (i + 1)) = 10 :=
sorry

end log_sum_geometric_sequence_l372_372816


namespace n_hyperplanes_cover_grid_l372_372772

theorem n_hyperplanes_cover_grid {n d : ℕ} (h1 : 3 ≤ n) (G : Set (Fin d → Fin n))
  (H : Finset (Set (Fin d → ℝ))) 
  (h2 : card G = n^d) 
  (h3 : card H = 2 * n - 3) 
  (h4 : ∀ p ∈ G, ∃ h ∈ H, p ∈ h) : 
  ∃ (H_sub : Finset (Set (Fin d → ℝ))), H_sub ⊆ H ∧ card H_sub = n ∧ ∀ p ∈ G, ∃ h ∈ H_sub, p ∈ h :=
sorry

end n_hyperplanes_cover_grid_l372_372772


namespace largest_unsatisfiable_group_l372_372887

theorem largest_unsatisfiable_group :
  ∃ n : ℕ, (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) ∧ (∀ m : ℕ, m > n → ∃ a b c : ℕ, m = 6 * a + 9 * b + 20 * c) ∧ n = 43 :=
by
  sorry

end largest_unsatisfiable_group_l372_372887


namespace geometric_progression_properties_l372_372502

-- Define the first term and the fifth term given
def b₁ := Real.sqrt 3
def b₅ := Real.sqrt 243

-- Define the nth term formula for geometric progression
def geometric_term (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

-- State both the common ratio and the sixth term
theorem geometric_progression_properties :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ 
           geometric_term b₁ q 5 = b₅ ∧ 
           geometric_term b₁ q 6 = 27 ∨ geometric_term b₁ q 6 = -27 :=
by
  sorry

end geometric_progression_properties_l372_372502


namespace bamboo_sections_length_l372_372779

variable {n d : ℕ} (a : ℕ → ℕ)
variable (h_arith : ∀ k, a (k + 1) = a k + d)
variable (h_top : a 1 = 10)
variable (h_sum_last_three : a n + a (n - 1) + a (n - 2) = 114)
variable (h_geom_6 : (a 6) ^ 2 = a 1 * a n)

theorem bamboo_sections_length : n = 16 := 
by 
  sorry

end bamboo_sections_length_l372_372779


namespace all_points_on_parabola_l372_372301

noncomputable def point_on_parabola (s : ℝ) : ℝ × ℝ :=
  (2^s - 3, 4^s - 7 * 2^s - 1)

theorem all_points_on_parabola : ∀ s : ℝ, ∃ x y : ℝ, point_on_parabola s = (x, y) ∧ y = x^2 - x - 13 :=
by
  intros s
  let x := 2^s - 3
  let y := 4^s - 7 * 2^s - 1
  use (x, y)
  split
  {
    unfold point_on_parabola,
    rfl
  }
  {
    calc y = (2^s)^2 - 7 * (2^s) - 1 : by rfl
    ... = (x + 3)^2 - 7 * (x + 3) - 1 : by rw [x]
    ... = x^2 + 6 * x + 9 - 7 * x - 21 - 1 : by ring
    ... = x^2 - x - 13 : by ring
  }
  sorry

end all_points_on_parabola_l372_372301


namespace largest_arithmetic_mean_of_two_digit_numbers_l372_372287

theorem largest_arithmetic_mean_of_two_digit_numbers 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100)
  (h3 : a > b)
  (h4 : (a + b : ℝ) / 2 = 25/24 * real.sqrt (a * b)) : (a + b) / 2 = 75 :=
sorry

end largest_arithmetic_mean_of_two_digit_numbers_l372_372287


namespace trapezoid_circle_tangent_ratio_l372_372147

/-- Given trapezoid EFGH with specified side lengths,
    where EF is parallel to GH, and a circle with
    center Q on EF tangent to FG and HE,
    the ratio EQ : QF is 12 : 37. -/
theorem trapezoid_circle_tangent_ratio :
  ∀ (EF FG GH HE : ℝ) (EQ QF : ℝ),
  EF = 40 → FG = 25 → GH = 12 → HE = 35 →
  ∃ (Q : ℝ) (EQ QF : ℝ),
  EQ + QF = EF ∧ EQ / QF = 12 / 37 ∧ gcd 12 37 = 1 :=
by
  sorry

end trapezoid_circle_tangent_ratio_l372_372147


namespace exists_x_for_log_eqn_l372_372720

theorem exists_x_for_log_eqn (a : ℝ) (ha : 0 < a) :
  ∃ (x : ℝ), (1 < x) ∧ (Real.log (a * x) / Real.log 10 = 2 * Real.log (x - 1) / Real.log 10) ∧ 
  x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 := sorry

end exists_x_for_log_eqn_l372_372720


namespace train_passes_pole_time_l372_372413

/-- Time in seconds for a train of length 200 meters traveling at 80 kmph to completely pass an electric pole is approximately 9 seconds. -/
theorem train_passes_pole_time :
  let distance := 200 -- meters
  let speed_kmph := 80 -- kmph
  let speed_mps := speed_kmph * 1000 / 3600 -- converting speed from kmph to m/s
  let time := distance / speed_mps -- in seconds
  (time ≈ 9) :=
by
  let distance := 200 -- meters
  let speed_kmph := 80 -- kmph
  let speed_mps := speed_kmph * 1000 / 3600 -- converting speed from kmph to m/s
  let time := distance / speed_mps -- in seconds
  have h1 : speed_mps = 22.22 := sorry
  have h2 : time = 200 / 22.22 := by rw h1; reflexivity
  have h3 : time ≈ 9 := sorry
  exact h3

end train_passes_pole_time_l372_372413


namespace complex_sum_real_part_l372_372979

theorem complex_sum_real_part :
  (1 / 2 ^ 1988) * ∑ n in Finset.range 995, (-3 : ℂ) ^ n * (Nat.choose 1988 (2 * n) : ℂ) = -Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_sum_real_part_l372_372979


namespace probability_product_is_minus_one_l372_372137

noncomputable def vertices : Finset ℂ := 
  { √2 * Complex.i, -√2 * Complex.i,
    (1 + Complex.i) / √8, (-1 + Complex.i) / √8,
    (1 - Complex.i) / √8, (-1 - Complex.i) / √8 }

noncomputable def chosen_vertices (n : ℕ) : Finset (Finset ℂ) :=
  (Finset.range n).powerset.bind (λ s, if s.card = 12 then {s.image λ i, (Finset.range n).choose i} else ∅)

theorem probability_product_is_minus_one :
  (chosen_vertices 12).card / (vertices.card^12) = 2^2 * 5 * 11 / 3^10 :=
sorry

end probability_product_is_minus_one_l372_372137


namespace sum_of_interior_diagonals_l372_372222

theorem sum_of_interior_diagonals (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 50) (h2 : x * y + y * z + z * x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 :=
by 
  sorry

end sum_of_interior_diagonals_l372_372222


namespace cyclic_quadrilateral_l372_372785

noncomputable def acute_triangle := Type

variables 
  (A B C A1 C1 P Q M H R : acute_triangle)
  (triangle_ABC : ∀ {A B C : acute_triangle}, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
                    (let α := (B - A), β := (C - B), γ := (A - C) in
                     α.x * β.y - α.y * β.x ≠ 0))
  (is_altitude : ∀ {A1 A B C : acute_triangle}, is_line (B, C) (A, A1) ∧ is_perpendicular (A, A1) (B, C))
  (is_angle_bisector : ∀ {P Q A B C A1 C1 : acute_triangle}, 
                        is_line (A, B) (Q, C) ∧ is_line (B, C) (P, A) ∧ 
                        ∠BPC = ∠BPQ)
  (midpoint_M : ∀ {A C M : acute_triangle}, is_midpoint M A C)
  (orthocenter : ∀ {A B C H : acute_triangle}, is_orthocenter H A B C)
  (bisector_intersect_HM : ∀ {A B C P Q M H R : acute_triangle}, 
                           is_line (A, B) (C, A1) ∧ is_line (B, C) (A, C1) ∧ 
                           ∠BPR = ∠BQR ∧ is_line (H, M) (R, bisector A B C (∠ABC)))

theorem cyclic_quadrilateral
  (A B C A1 C1 P Q M H R : acute_triangle)
  (triangle_ABC : ∀ {A B C : acute_triangle}, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
                    (let α := (B - A), β := (C - B), γ := (A - C) in
                     α.x * β.y - α.y * β.x ≠ 0))
  (is_altitude_A : ∀ {A1 A B C : acute_triangle}, is_line (B, C) (A, A1) ∧ is_perpendicular (A, A1) (B, C))
  (is_altitude_C : ∀ {C1 C A B : acute_triangle}, is_line (A, B) (C, C1) ∧ is_perpendicular (C, C1) (A, B))
  (is_angle_bisector_P : ∀ {P A B C : acute_triangle}, is_line (B, C) (P, A) ∧ P ∈ angle_bisector (∠BAC C A1))
  (is_angle_bisector_Q : ∀ {Q A B C : acute_triangle}, is_line (A, B) (Q, C) ∧ Q ∈ angle_bisector (∠BCA A C1))
  (midpoint_M : ∀ {A C M : acute_triangle}, is_midpoint M A C)
  (orthocenter_H : ∀ {A B C H : acute_triangle}, is_orthocenter H A B C)
  (bisector_intersect_HM : ∀ {R M H B : acute_triangle}, R ∈ segment H M ∧ R ∈ angle_bisector (∠ABC AB BC)) :
  cyclic R P B Q :=
by
  sorry

end cyclic_quadrilateral_l372_372785


namespace harry_took_5_eggs_l372_372141

theorem harry_took_5_eggs (initial : ℕ) (left : ℕ) (took : ℕ) 
  (h1 : initial = 47) (h2 : left = 42) (h3 : left = initial - took) : 
  took = 5 :=
sorry

end harry_took_5_eggs_l372_372141


namespace stddev_proof_l372_372705

-- The given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- The five numbers given the solutions of the quadratic equation
def numbers (a b : ℝ) : List ℝ := [3, a, 4, b, 5]

-- Function to calculate the mean of a list of numbers
def mean (l : List ℝ) : ℝ := (l.sum) / l.length

-- Function to calculate the variance of a list of numbers
def variance (l : List ℝ) : ℝ := 
  let m := mean l
  (l.map (λ x, (x - m) ^ 2)).sum / l.length

-- Function to calculate the standard deviation of a list of numbers
def stddev (l : List ℝ) : ℝ := Real.sqrt (variance l)

-- Statement to be proven
theorem stddev_proof :
  ∀ (a b : ℝ), quadratic_eq a → quadratic_eq b → a ≠ b → stddev (numbers a b) = Real.sqrt 2 :=
by
  sorry

end stddev_proof_l372_372705


namespace find_n_l372_372291

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [MOD 15] ∧ n = 10 := by
  use 10
  repeat { sorry }

end find_n_l372_372291


namespace martingale_uncorrelated_increments_l372_372561

open ProbabilityTheory

variables {Ω : Type*} {F : filtration ℝ} [measure_space Ω]
  (ξ : nat → Ω → ℝ)
  {a b c d : ℕ}

noncomputable def is_martingale (X : nat → Ω → ℝ) (ℱ : filtration ℝ) : Prop :=
  ∀ (n : ℕ), X n =𝔣[ℱ n] X (n + 1)

theorem martingale_uncorrelated_increments {a b c d : ℕ} (h_mart : is_martingale ξ F)
  (h : a < b) (h2 : b < c) (h3 : c < d) :
  cov (λ ω, ξ d ω - ξ c ω) (λ ω, ξ b ω - ξ a ω) = 0 :=
sorry

end martingale_uncorrelated_increments_l372_372561


namespace absolute_value_z_squared_minus_2z_l372_372757

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem
theorem absolute_value_z_squared_minus_2z : complex.abs (z^2 - 2*z) = 2 := by
  sorry

end absolute_value_z_squared_minus_2z_l372_372757


namespace problem_f_n_binomial_l372_372686

open BigOperators

theorem problem_f_n_binomial (n : ℕ) (h : 0 < n) : 
  let f := λ (n : ℕ), (∑ k in Finset.range n, (k + 1) * Nat.choose n k * Nat.choose n (k + 1)) / n in
  f n = Nat.choose (2 * n - 1) n := 
by
  sorry

end problem_f_n_binomial_l372_372686


namespace sin_R_l372_372394

variables {P Q R : Type} 
variables [Inhabited P] [Inhabited Q] [Inhabited R]

def right_triangle (P Q R : Type) : Prop := sorry -- definition of RightTriangle

theorem sin_R (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
  (h : right_triangle P Q R) (hQ : ∠Q = 90) (hP : sin ∠P = 3 / 5) :
  sin ∠R = 4 / 5 := by
  sorry

end sin_R_l372_372394


namespace inequality_of_positive_numbers_l372_372824

variable {a b : ℝ}

theorem inequality_of_positive_numbers (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ sqrt(ab) + sqrt((a^2 + b^2) / 2) :=
sorry

end inequality_of_positive_numbers_l372_372824


namespace total_wicks_20_l372_372961

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l372_372961


namespace sally_eats_sandwiches_l372_372844

theorem sally_eats_sandwiches
  (saturday_sandwiches : ℕ)
  (bread_per_sandwich : ℕ)
  (total_bread : ℕ)
  (one_sandwich_on_sunday : ℕ)
  (saturday_bread : saturday_sandwiches * bread_per_sandwich = 4)
  (total_bread_consumed : total_bread = 6)
  (bread_on_sundy : bread_per_sandwich = 2) :
  (total_bread - saturday_sandwiches * bread_per_sandwich) / bread_per_sandwich = one_sandwich_on_sunday :=
sorry

end sally_eats_sandwiches_l372_372844


namespace median_salary_l372_372598

theorem median_salary 
    (CEO_count : ℕ) (SeniorManager_count : ℕ) (Manager_count : ℕ) 
    (AssistantManager_count : ℕ) (Clerk_count : ℕ) 
    (CEO_salary : ℕ) (SeniorManager_salary : ℕ) 
    (Manager_salary : ℕ) (AssistantManager_salary : ℕ) (Clerk_salary : ℕ)
    (total_employees : ℕ)
    (median_salary : ℕ)
    (h1 : CEO_count = 1)
    (h2 : SeniorManager_count = 4)
    (h3 : Manager_count = 15)
    (h4 : AssistantManager_count = 20)
    (h5 : Clerk_count = 40)
    (h6 : CEO_salary = 150000)
    (h7 : SeniorManager_salary = 95000)
    (h8 : Manager_salary = 70000)
    (h9 : AssistantManager_salary = 45000)
    (h10 : Clerk_salary = 18000)
    (h_total_employees : total_employees = CEO_count + SeniorManager_count + Manager_count + AssistantManager_count + Clerk_count)
    (h_median_salary_at_40th_and_41st : median_salary = AssistantManager_salary) :
    median_salary = 45000 := 
by
  rw [←h_median_salary_at_40th_and_41st]
  exact h9

end median_salary_l372_372598


namespace complex_multiplication_imaginary_zero_l372_372452

noncomputable def is_real (z : ℂ) : Prop :=
  z.im = 0

theorem complex_multiplication_imaginary_zero (x : ℝ) :
  let z1 := (1 : ℂ) + (1 : ℂ) * complex.i
  let z2 := (2 : ℂ) + x * complex.i
  is_real (z1 * z2) ↔ x = -2 := 
by
  sorry

end complex_multiplication_imaginary_zero_l372_372452


namespace polar_equiv_rectangular_max_value_pn_pm_l372_372404

noncomputable def polar_to_rectangular (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * real.cos θ - 6 * ρ * real.sin θ + 12 = 0

noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

theorem polar_equiv_rectangular :
  ∀ (ρ θ : ℝ), polar_to_rectangular ρ θ ↔ curve_rectangular (ρ * real.cos θ) (ρ * real.sin θ) :=
by sorry

/-- A proof that the maximum value of |PM| + |PN| for points P on the curve 
  $(x - 2)^2 + (y - 3)^2 = 1$ is 6 + sqrt 2, with perpendicular foot M, N on the line x = -1. -/
theorem max_value_pn_pm :
  ∀ (P : ℝ × ℝ) (M N : ℝ × ℝ),
    curve_rectangular P.1 P.2 → 
    M.1 = -1 ∧ N.1 = -1 → 
    12 - 6 ≤ |P.2 + 3| + |P.2 + 1| ∧ 12 - 6 ≥ |P.2 + 3| + |P.2 + 1| :=
by {
  sorry -- Proof to be filled in
}

end polar_equiv_rectangular_max_value_pn_pm_l372_372404


namespace entire_surface_area_l372_372596

noncomputable def original_cube_side : ℝ := 5
noncomputable def large_hole_side : ℝ := 2
noncomputable def small_hole_side : ℝ := 0.5

theorem entire_surface_area (original_cube_side large_hole_side small_hole_side : ℝ)
  (h_original_cube : original_cube_side = 5)
  (h_large_hole : large_hole_side = 2)
  (h_small_hole : small_hole_side = 0.5)
  : 
  let original_surface_area := 6 * (original_cube_side ^ 2),
      removed_large_hole := 6 * (large_hole_side ^ 2),
      exposed_by_large_hole := 6 * 4 * (large_hole_side ^ 2),
      removed_small_hole := 6 * 4 * (small_hole_side ^ 2),
      total_surface_area := original_surface_area - removed_large_hole + exposed_by_large_hole + removed_small_hole
  in total_surface_area = 228 := 
by
  sorry

end entire_surface_area_l372_372596


namespace min_performances_l372_372199

theorem min_performances (n_pairs_per_show m n_singers : ℕ) (h1 : n_singers = 8) (h2 : n_pairs_per_show = 6) 
  (condition : 6 * m = 28 * 3) : m = 14 :=
by
  -- Use the assumptions to prove the statement
  sorry

end min_performances_l372_372199


namespace distance_covered_by_walk_l372_372583

theorem distance_covered_by_walk (
  (total_distance : ℝ) (train_fraction : ℝ) (bus_fraction : ℝ) (walk_fraction : ℝ)) :
  total_distance = 129.9999999999999 →                        
  train_fraction = 3/5 →
  bus_fraction = 7/20 →
  walk_fraction = 1 - train_fraction - bus_fraction →
  total_distance * walk_fraction = 6.499999999999991 :=
by 
  intros h1 h2 h3 h4
  sorry

end distance_covered_by_walk_l372_372583


namespace midpoint_of_segment_formed_by_perpendicular_bisectors_l372_372578

variables {A B C M O X Y Z : Point}
variables {l : Line}

-- Definitions involved in the problem
def is_median (M : Point) (A B C : Point) := 
∃ P Q : Point, M = midpoint P Q ∧ P = A ∧ Q = B ∧ line_through B C

def is_perpendicular_to_median (l : Line) (M : Point) :=
perpendicular l (line_through M B)

def intersects_at (P : Point) (l : Line) :=
intersection_points P l

def intersect_line_with_perpendicular_bisectors (l : Line) (P Q R S : Point) :=
∃ X Y Z : Point, 
intersects_at X l ∧
intersects_at Y l ∧
intersects_at Z l ∧
is_perpendicular_bisector X Q ∧
is_perpendicular_bisector Y R ∧
is_perpendicular_bisector Z S

def is_midpoint (Y : Point) (X Z : Point) :=
midpoint Y X Z

theorem midpoint_of_segment_formed_by_perpendicular_bisectors
  (A B C M O X Y Z : Point)
  (l : Line) :
  is_median M A B C →
  is_perpendicular_to_median l M →
  intersect_line_with_perpendicular_bisectors l A B C (midpoint A C) →
  is_midpoint Y X Z := 
sorry

end midpoint_of_segment_formed_by_perpendicular_bisectors_l372_372578


namespace full_price_ticket_revenue_correct_l372_372932

-- Define the constants and assumptions
variables (f t : ℕ) (p : ℝ)

-- Total number of tickets sold
def total_tickets := (f + t = 180)

-- Total revenue from ticket sales
def total_revenue := (f * p + t * (p / 3) = 2600)

-- Full price ticket revenue
def full_price_revenue := (f * p = 975)

-- The theorem combines the above conditions to prove the correct revenue from full-price tickets
theorem full_price_ticket_revenue_correct :
  total_tickets f t →
  total_revenue f t p →
  full_price_revenue f p :=
by
  sorry

end full_price_ticket_revenue_correct_l372_372932


namespace find_length_DY_l372_372848

noncomputable def length_DY : Real :=
    let AE := 2
    let AY := 4 * AE
    let DY  := Real.sqrt (66 + Real.sqrt 5)
    DY

theorem find_length_DY : length_DY = Real.sqrt (66 + Real.sqrt 5) := 
  by
    sorry

end find_length_DY_l372_372848


namespace calc_f_y_eq_2f_x_l372_372371

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem calc_f_y_eq_2f_x (x : ℝ) (h : -1 < x) (h' : x < 1) :
  f ( (2 * x + x^2) / (1 + 2 * x^2) ) = 2 * f x := by
  sorry

end calc_f_y_eq_2f_x_l372_372371


namespace ellipse_eccentricity_a_l372_372014

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372014


namespace sum_squares_reciprocal_roots_l372_372101

open Complex

noncomputable def polynomial : ℂ → ℂ :=
  λ z, z^5 + a * z^4 + b * z^3 + c * z^2 + d * z + e

theorem sum_squares_reciprocal_roots (a b c d e : ℝ) (z : ℂ → ℂ)
  (roots : Multiset ℂ) (h_roots : roots.card = 5)
  (h_poly_roots : ∀ x ∈ roots, polynomial x = 0)
  (h_root_norms : ∀ x ∈ roots, |x| = 2) :
  (roots.map (λ z, (1/z)^2)).sum = 5 :=
by sorry

end sum_squares_reciprocal_roots_l372_372101


namespace mappings_A_to_B_mappings_B_to_A_l372_372451

def A := {a, b : Prop}
def B := {1, -1, 0 : Integer}

theorem mappings_A_to_B : (cardinality of all functions (A → B)) = 9 := 
  sorry

theorem mappings_B_to_A : (cardinality of all functions (B → A)) = 8 := 
  sorry

end mappings_A_to_B_mappings_B_to_A_l372_372451


namespace value_of_a3_l372_372722

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 1
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem value_of_a3 : a 3 = 10 := by
  sorry

end value_of_a3_l372_372722


namespace sum_binom_expr_eq_neg_half_l372_372985

def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else (n.factorial) / (k.factorial * (n - k).factorial)

noncomputable def sum_binom_expr : ℝ :=
  (1 / 2 ^ 1988) * ∑ n in Finset.range (995), (-3 : ℝ) ^ n * binom 1988 (2 * n)

theorem sum_binom_expr_eq_neg_half : sum_binom_expr = -1 / 2 := by
  sorry

end sum_binom_expr_eq_neg_half_l372_372985


namespace find_lambda_l372_372363

-- Define vectors a and b
def a : ℝ × ℝ := (-3, 2)
def b (λ : ℝ) : ℝ × ℝ := (4, λ)

-- Define the vectors in question
def v1 (λ : ℝ) := (a.1 + 3 * (b λ).1, a.2 + 3 * (b λ).2)
def v2 (λ : ℝ) := (2 * a.1 - (b λ).1, 2 * a.2 - (b λ).2)

-- Define the parallel condition
def parallel (x y : ℝ × ℝ) : Prop := ∃ k : ℝ, x = (k * y.1, k * y.2)

-- Prove that λ = -8/3 satisfies the parallel condition
theorem find_lambda : parallel (v1 (-8/3)) (v2 (-8/3)) :=
sorry

end find_lambda_l372_372363


namespace find_locus_of_p_l372_372102

noncomputable def locus_of_point_p (a b : ℝ) : Set (ℝ × ℝ) :=
{p | (p.snd = 0 ∧ -a < p.fst ∧ p.fst < a) ∨ (p.fst = a^2 / Real.sqrt (a^2 + b^2))}

theorem find_locus_of_p (a b : ℝ) (P : ℝ × ℝ) :
  (∃ (x0 y0: ℝ),
      P = (x0, y0) ∧
      ( ∃ (x1 y1 x2 y2 : ℝ),
        (x0 ≠ 0 ∨ y0 ≠ 0) ∧
        (x1 ≠ x2 ∨ y1 ≠ y2) ∧
        (y0 = 0 ∨ (b^2 * x0 = -a^2 * (x0 - Real.sqrt (a^2 + b^2)))) ∧
        ((y0 = 0 ∧ -a < x0 ∧ x0 < a) ∨ x0 = a^2 / Real.sqrt (a^2 + b^2)))) ↔ 
  P ∈ locus_of_point_p a b :=
sorry

end find_locus_of_p_l372_372102


namespace square_pyramid_sum_l372_372544

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end square_pyramid_sum_l372_372544


namespace distinct_patterns_4x4_grid_l372_372365

theorem distinct_patterns_4x4_grid : 
  ∃! (n : ℕ), (n = 10) ∧ 
  (∀ (grid : Fin (4) × Fin (4) → Bool), 
  (∃ (patterns : Finset (Fin 4 × Fin 4)), 
    patterns.card = 3 ∧ 
    ∀ (p1 p2 : Finset (Fin 4 × Fin 4)), 
    (p1 = sym_rot_flip p2 → p1 = p2)) 
  → n = patterns.card) :=
sorry

end distinct_patterns_4x4_grid_l372_372365


namespace trigonometric_identity_l372_372324

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
sorry

end trigonometric_identity_l372_372324


namespace twenty_paise_coins_l372_372190

/-- The total value of the coins in Rs -/
def coin_value (x y : ℕ) : ℝ := 0.20 * x + 0.25 * y

theorem twenty_paise_coins (x y : ℕ) (h1 : x + y = 342) (h2 : coin_value x y = 71) : x = 290 :=
by
  have h3 : 0.20 * x + 0.25 * y = 71 := h2
  have h4 : 0.20 * (x + y) = 0.20 * 342 := by
    rw [h1]
  rw [mul_add, mul_add] at h4
  have h5 : 0.20 * x + 0.20 * y = 68.4 := h4
  have h6 : 0.20 * x + 0.25 * y - (0.20 * x + 0.20 * y) = 71 - 68.4 := by
    rw [h3, h5]
  have h7 : 0.05 * y = 2.6 := by
    linarith
  have h8 : y = 2.6 / 0.05 := by
    field_simp
  have h9 : y = 52 := by
    rw div_eq_mul_inv at h8
    simp at h8
    rw inv_eq_one_div at h8
    norm_num at h8
  have h10 : x = 342 - y := by
    linarith
  show x = 290 from by
    subst h9
    norm_num at h10

end twenty_paise_coins_l372_372190


namespace find_primes_l372_372641

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end find_primes_l372_372641


namespace ali_wins_if_n_even_l372_372606

-- Define the conditions and the goal
def game_strategy (n : ℕ) (h : n ≥ 2) : Prop :=
  ∀ (P : polynomial ℚ), (∃ r : ℚ, P.eval r = 0) ↔ even n

theorem ali_wins_if_n_even {n : ℕ} (h : n ≥ 2) :
  game_strategy n h :=
sorry

end ali_wins_if_n_even_l372_372606


namespace quadrilateral_PQRS_area_l372_372403

-- Definitions based on given conditions
variables (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
variables (PQ QR RS PS QS : ℝ)

-- Conditions
axiom h1 : QR = 5
axiom h2 : RS = 12
axiom h3 : angle R Q S = 90

-- Definition of right angle
noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  1/2 * a * b

-- Proof statement
theorem quadrilateral_PQRS_area : right_triangle_area QR RS = 30 :=
begin
  rw [right_triangle_area, h1, h2],
  norm_num,
end

end quadrilateral_PQRS_area_l372_372403


namespace sum_cos_4x_4y_4z_l372_372821

theorem sum_cos_4x_4y_4z (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 :=
by
  sorry

end sum_cos_4x_4y_4z_l372_372821


namespace probability_no_snow_l372_372511

theorem probability_no_snow (p_snow : ℚ) (h : p_snow = 4/5) : 
  let p_no_snow := 1 - p_snow in
  p_no_snow ^ 5 = 1 / 3125 := 
by
  sorry

end probability_no_snow_l372_372511


namespace coefficient_x6y2_expansion_l372_372113

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem coefficient_x6y2_expansion :
  (∑ r in Finset.range 6, binomial 5 r * binomial (5 - r) 0 * 2^r *
     (x^((5 - r) * 2) * y^(r + 0))) = 40 :=
sorry

end coefficient_x6y2_expansion_l372_372113


namespace variance_of_data_set_l372_372951

theorem variance_of_data_set :
  let data := [10, 6, 8, 5, 6]
  let n := data.length
  let mean := (data.sum) / n
  let variance := (data.map (λ x => ((x.toFraction - mean) ^ 2).toRational)).sum / n
  variance = 16 / 5 :=
by
  sorry

end variance_of_data_set_l372_372951


namespace ellipse_eccentricity_l372_372046

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372046


namespace inverse_variation_function_l372_372967

theorem inverse_variation_function (k : ℝ) (O A B C E F : Point)
  (hO : O = (0, 0)) (hA : A = (6, 0)) (hB : B = (6, 5)) (hC : C = (0, 5))
  (hE : E = (6, k / 6)) (hF : F = (k / 5, 5)) 
  (hCond : CF < FB)
  (hArea : area O E F - area B F E = 5 + 11 / 30) :
  k = 7 :=
sorry

end inverse_variation_function_l372_372967


namespace remove_min_elements_such_that_product_condition_l372_372683

theorem remove_min_elements_such_that_product_condition :
  ∃ R : set ℕ, (∀ x ∈ R, x ≤ 1982) ∧
  (∀ a b, a ∈ R → b ∈ R → a ≠ b → a * b ∉ R) ∧
  (|({n ∈ (finset.range 1983) | n ≤ 1982} \ R)| = 43) := sorry

end remove_min_elements_such_that_product_condition_l372_372683


namespace trapezoid_other_base_possible_lengths_l372_372563

-- Definition of the trapezoid problem.
structure Trapezoid where
  height : ℕ
  leg1 : ℕ
  leg2 : ℕ
  base1 : ℕ

-- The given conditions
def trapezoid_data : Trapezoid :=
{ height := 12, leg1 := 20, leg2 := 15, base1 := 42 }

-- The proof problem in Lean 4 statement
theorem trapezoid_other_base_possible_lengths (t : Trapezoid) :
  t = trapezoid_data → (∃ b : ℕ, (b = 17 ∨ b = 35)) :=
by
  intro h_data_eq
  sorry

end trapezoid_other_base_possible_lengths_l372_372563


namespace problem_1_problem_2_l372_372715

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log (x + 1) + Real.log (1 - x) + a * (x + 1)

def mono_intervals (a : ℝ) : Set ℝ × Set ℝ := 
  if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) 
  else (∅, ∅)

theorem problem_1 (a : ℝ) (h_pos : a > 0) : 
  mono_intervals a = if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) else (∅, ∅) :=
sorry

theorem problem_2 (h_max : f a 0 = 1) (h_pos : a > 0) : 
  a = 1 :=
sorry

end problem_1_problem_2_l372_372715


namespace calculate_total_interest_rate_l372_372952

noncomputable def total_investment : ℝ := 10000
noncomputable def amount_invested_11_percent : ℝ := 3750
noncomputable def amount_invested_9_percent : ℝ := total_investment - amount_invested_11_percent
noncomputable def interest_rate_9_percent : ℝ := 0.09
noncomputable def interest_rate_11_percent : ℝ := 0.11

noncomputable def interest_from_9_percent : ℝ := interest_rate_9_percent * amount_invested_9_percent
noncomputable def interest_from_11_percent : ℝ := interest_rate_11_percent * amount_invested_11_percent

noncomputable def total_interest : ℝ := interest_from_9_percent + interest_from_11_percent

noncomputable def total_interest_rate : ℝ := (total_interest / total_investment) * 100

theorem calculate_total_interest_rate :
  total_interest_rate = 9.75 :=
by 
  sorry

end calculate_total_interest_rate_l372_372952


namespace matrix_det_example_l372_372618

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l372_372618


namespace solve_quadratic_eq_l372_372095

-- Define the equation as part of the conditions
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 + 4 * x - 1 = 0

theorem solve_quadratic_eq (x : ℝ) :
  quadratic_eq x ↔ (x = -1 + sqrt 6 / 2 ∨ x = -1 - sqrt 6 / 2) :=
by sorry

end solve_quadratic_eq_l372_372095


namespace expand_expression_l372_372637

variable {R : Type} [CommRing R]
variables (x y : R)

theorem expand_expression :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 :=
by
  sorry

end expand_expression_l372_372637


namespace arithmetic_seq_inequality_l372_372300

variable {a : ℕ → ℝ}  -- the arithmetic sequence
variable {S : ℕ → ℝ}  -- the sum of first n terms

-- S_n is the sum of the first n terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := S n = ∑ i in range (n + 1), a i

-- the condition that the product (S8 - S5)(S8 - S4) is less than zero
def condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := (S 8 - S 5) * (S 8 - S 4) < 0

-- the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n m : ℕ, a (n + m) = a n + a m

theorem arithmetic_seq_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : sum_of_first_n_terms a 8)
  (h2 : sum_of_first_n_terms a 5)
  (h3 : sum_of_first_n_terms a 4)
  (cond : condition a S)
  (arith_seq : arithmetic_sequence a)
  : |a 6| > |a 7| :=
sorry

end arithmetic_seq_inequality_l372_372300


namespace hexagonal_prism_edge_length_sum_l372_372111

theorem hexagonal_prism_edge_length_sum (side_length height : ℕ) 
    (h₁ : side_length = 6) (h₂ : height = 11) : 
    (12 * side_length + 6 * height) = 138 :=
by
  rw [h₁, h₂]
  sorry

end hexagonal_prism_edge_length_sum_l372_372111


namespace product_of_values_t_squared_eq_49_l372_372652

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372652


namespace area_TUW_is_0_75_l372_372393

-- Assigning points
def P : (ℝ × ℝ) := (0, 0)
def Q : (ℝ × ℝ) := (2, 0)
def R : (ℝ × ℝ) := (2, 3)
def S : (ℝ × ℝ) := (0, 3)

-- Midpoints
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def T : (ℝ × ℝ) := midpoint Q R
def U : (ℝ × ℝ) := midpoint R S
def V : (ℝ × ℝ) := midpoint S P
def W : (ℝ × ℝ) := midpoint V T

-- Area calculation function
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Lean statement to prove the area of the triangle TUW is 0.75
theorem area_TUW_is_0_75 : triangle_area T U W = 0.75 :=
  sorry

end area_TUW_is_0_75_l372_372393


namespace find_a_l372_372016

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372016


namespace pow_mul_eq_given_problem_l372_372134

theorem pow_mul_eq (a : ℝ) (b c : ℝ) : a^(b + c) = a^b * a^c :=
by sorry

theorem given_problem : (625 : ℝ) ^ 0.12 * (625 : ℝ) ^ 0.38 = 25 :=
by {
  -- Proof goes here
  sorry
}

end pow_mul_eq_given_problem_l372_372134


namespace exists_disk_of_radius_one_containing_half_points_l372_372535

/-- Given n points in a plane such that among any triplet of points, two of them are at a distance of less than 1, there exists a disk of radius 1 containing at least floor(n/2) of these points. -/
theorem exists_disk_of_radius_one_containing_half_points (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    dist (points i) (points j) < 1 ∨ dist (points j) (points k) < 1 ∨ dist (points k) (points i) < 1) :
  ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1 ∧ (↑(Finset.univ.filter (λ i, dist c (points i) ≤ r)).card ≥ n / 2) :=
sorry

end exists_disk_of_radius_one_containing_half_points_l372_372535


namespace distance_between_first_and_last_tree_l372_372849

theorem distance_between_first_and_last_tree
  (n : ℕ)
  (trees : ℕ)
  (dist_between_first_and_fourth : ℕ)
  (eq_dist : ℕ):
  trees = 6 ∧ dist_between_first_and_fourth = 60 ∧ eq_dist = dist_between_first_and_fourth / 3 ∧ n = (trees - 1) * eq_dist → n = 100 :=
by
  intro h
  sorry

end distance_between_first_and_last_tree_l372_372849


namespace surface_area_increase_l372_372191

-- Define the original length of the cube's edge
variable (L : ℝ) (hL : 0 < L)

-- Define the increase percentage
def percent_increase : ℝ := 0.30

-- Define the new edge length after the increase
def new_edge_length : ℝ := (1 + percent_increase) * L

-- Define the original surface area
def original_surface_area : ℝ := 6 * L^2

-- Define the new surface area
def new_surface_area : ℝ := 6 * new_edge_length^2

-- Define the percentage increase in surface area
def percentage_increase_area : ℝ :=
  ((new_surface_area - original_surface_area) / original_surface_area) * 100

-- The theorem to be proved
theorem surface_area_increase (L_pos : 0 < L) : percentage_increase_area L hL = 69 := by
  sorry


end surface_area_increase_l372_372191


namespace y_gt_1_l372_372123

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end y_gt_1_l372_372123


namespace smallest_positive_angle_l372_372988

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ 4 * sin y * cos y^3 - 4 * sin y^3 * cos y = cos y ∧ y = 18 :=
by
  sorry

end smallest_positive_angle_l372_372988


namespace inequality_proof_l372_372825

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ real.sqrt (a * b) + real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l372_372825


namespace color_of_last_bead_l372_372797

noncomputable def length_of_pattern : ℕ := 9

def bead_pattern : list string :=
  ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def first_bead : string := "red"

def total_beads : ℕ := 84

theorem color_of_last_bead (h_pattern_length : length_of_pattern = 9)
    (h_total_beads : total_beads = 84) (h_first_bead : first_bead = "red")
    (h_pattern : bead_pattern = ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]) :
  bead_pattern[((total_beads - 1) % length_of_pattern)] = "orange" :=
by
  sorry

end color_of_last_bead_l372_372797


namespace angle_bisector_length_of_angle_B_l372_372082

variables {R : Type*} [LinearOrder R] [Field R] [FloorRing R]
variables (A B C I O K : R)
variables (OI IK AK : R)

-- Given conditions
axiom I_is_incenter : ∀ (Δ : Triangle (ℝ)), Is_incenter I Δ
axiom O_is_excenter : ∀ (Δ : Triangle (ℝ)), Is_excenter O Δ AC
axiom AC_intersects_OI_at_K : ∀ (Δ : Triangle (ℝ)), Intersect AC (Line_through O I) K
axiom OI_length : OI = 50
axiom IK_length : IK = 18
axiom AK_length : AK = 24

-- To be proved
theorem angle_bisector_length_of_angle_B (Δ : Triangle (ℝ)) :
  ∠ B = length_of_angle_bisector Δ B = 576 / 7 :=
sorry

end angle_bisector_length_of_angle_B_l372_372082


namespace geometric_progression_sixth_term_proof_l372_372499

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l372_372499


namespace range_of_f_l372_372327

noncomputable def f (x : ℝ) (k n : ℝ) : ℝ := x^(k/n)

theorem range_of_f (k n : ℝ) (h1 : k < 0) (h2 : n > 0) :
  set.range (λ x : ℝ, if x ≥ 2 then f x k n else 0) = set.Ioc 0 (2^(k/n)) ∪ {0} :=
by
  sorry

end range_of_f_l372_372327


namespace find_a_l372_372343

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x
noncomputable def g (a x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem find_a (x₁ x₂ a : ℝ) (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (hf : f a x₁ + f a x₂ = -4) : a = 4 :=
sorry

end find_a_l372_372343


namespace find_n_l372_372532

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 14 = 56) (h_gcf : Nat.gcd n 14 = 12) : n = 48 :=
by
  sorry

end find_n_l372_372532


namespace determinant_example_l372_372615

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l372_372615


namespace angle_A_is_60_l372_372381

noncomputable def angle_A {A B C : Type*} [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]
  (AB BC AC : ℝ)
  (h1 : AB = 3)
  (h2 : BC = Real.sqrt 13)
  (h3 : AC = 4) : ℝ :=
let c := AB in
let a := BC in
let b := AC in
let cosA := (b^2 + c^2 - a^2) / (2 * b * c) in
Real.arccos (cosA)

theorem angle_A_is_60 {A B C : Type*} [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]
  (AB BC AC : ℝ)
  (h1 : AB = 3)
  (h2 : BC = Real.sqrt 13)
  (h3 : AC = 4) : angle_A AB BC AC h1 h2 h3 = π / 3 :=
by
  -- Proof can be added here to convert Lean statement into a provable statement
  sorry

end angle_A_is_60_l372_372381


namespace u_1000_eq_2036_l372_372440

open Nat

def sequence_term (n : ℕ) : ℕ :=
  let sum_to (k : ℕ) := k * (k + 1) / 2
  if n ≤ 0 then 0 else
  let group := (Nat.sqrt (2 * n)) + 1
  let k := n - sum_to (group - 1)
  (group * group) + 4 * (k - 1) - (group % 4)

theorem u_1000_eq_2036 : sequence_term 1000 = 2036 := sorry

end u_1000_eq_2036_l372_372440


namespace true_contrapositive_of_right_triangle_acute_angles_complementary_l372_372231

theorem true_contrapositive_of_right_triangle_acute_angles_complementary :
  (∀ (α β : ℝ), is_right_triangle α β → is_complementary α β) :=
by
  sorry

structure is_right_triangle (α β γ : ℝ) : Prop :=
  (right_angle : γ = 90)
  (acute_angle_sum : α + β = 90)

def is_complementary (α β : ℝ) : Prop :=
  α + β = 90

example : (∀ α β γ : ℝ, is_right_triangle α β γ → is_complementary α β) :=
by
  intros α β γ h
  cases h with ha hb
  exact hb

end true_contrapositive_of_right_triangle_acute_angles_complementary_l372_372231


namespace solve_for_x_l372_372759

theorem solve_for_x (x : ℝ) (h : x + real.sqrt 81 = 25) : x = 16 :=
by
  sorry

end solve_for_x_l372_372759


namespace abs_z2_minus_2z_eq_2_l372_372752

theorem abs_z2_minus_2z_eq_2 (z : ℂ) (h : z = 1 + complex.I) : abs (z^2 - 2*z) = 2 := by
  sorry

end abs_z2_minus_2z_eq_2_l372_372752


namespace max_value_ab_ac_bc_exists_l372_372448

noncomputable def max_value_ab_ac_bc (a b c : ℝ) (h : 2 * a + 3 * b + c = 6) : ℝ :=
  ab + ac + bc

theorem max_value_ab_ac_bc_exists 
  (a b c : ℝ)
  (h : 2 * a + 3 * b + c = 6) 
  : ∃ a b c, ab + ac + bc = 3/2 * a^2 - 4 * a * b - 3 * b^2 + 6 * a + 6 * b := sorry

end max_value_ab_ac_bc_exists_l372_372448


namespace sum_impossible_l372_372900

def is_complementary_digits (a b : ℕ) : Prop :=
  (a + b = 9)

def has_complementary_digit_pairs (A B : ℕ) : Prop :=
  ∀ i, let a_i := (A / 10^i) % 10 in
       let b_i := (B / 10^i) % 10 in
       is_complementary_digits a_i b_i

theorem sum_impossible (A B : ℕ) (h1 : A ≠ B) (h2 : A + B = 10^1999 - 1) (h3 : has_complementary_digit_pairs A B) :
  false :=
sorry

end sum_impossible_l372_372900


namespace base_of_number_l372_372214

theorem base_of_number (b : ℕ) (h : (b ^ 300).digits = 91) : b = 2 :=
by
  sorry

end base_of_number_l372_372214


namespace product_of_values_t_squared_eq_49_l372_372650

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372650


namespace sarah_age_l372_372091

-- Defining the ages of individuals using variable names
variable (A : ℝ) -- Ana's age (predefined as 15)
variable (B : ℝ) -- Billy's age
variable (M : ℝ) -- Mark's age
variable (S : ℝ) -- Sarah's age

-- Stating the conditions
def condition_1 : Prop := S = 3 * M - 4
def condition_2 : Prop := M = B + 4
def condition_3 : Prop := B = A / 2
def condition_4 : Prop := A = 15

-- The theorem to be proven
theorem sarah_age : A = 15 → S = 30.5 :=
by
  -- Immediate definition of A for context
  assume hA : A = 15,
  -- Using conditions to deduce B, then M, then S
  let B' := A / 2,
  let M' := B' + 4,
  let S' := 3 * M' - 4,
  -- Show that substituting A = 15 gives S' = 30.5
  calc
  S' = 3 * ((15 / 2) + 4) - 4 : by simp [hA, condition_3, condition_2, condition_1]
  ... = 30.5 : by norm_num

-- Adding sorry to skip the actual proof execution
sorry

end sarah_age_l372_372091


namespace find_coins_l372_372138

-- Definitions based on conditions
structure Wallet where
  coin1 : ℕ
  coin2 : ℕ
  h_total_value : coin1 + coin2 = 15
  h_not_five : coin1 ≠ 5 ∨ coin2 ≠ 5

-- Theorem statement based on the proof problem
theorem find_coins (w : Wallet) : (w.coin1 = 5 ∧ w.coin2 = 10) ∨ (w.coin1 = 10 ∧ w.coin2 = 5) := by
  sorry

end find_coins_l372_372138


namespace range_of_m_l372_372701

theorem range_of_m (p q : Prop) (m x : ℝ)
  (h1 : p ↔ (x - m)^2 > 3 * (x - m))
  (h2 : q ↔ x^2 + 3*x - 4 < 0)
  (h3 : ∀ x, q x → p x) :
  m ≥ 1 ∨ m ≤ -7 :=
by
  sorry

end range_of_m_l372_372701


namespace solve_quadratic_l372_372097

theorem solve_quadratic : 
  (∃ x₁ x₂ : ℝ, x₁ = -1 + sqrt 6 / 2 ∧ x₂ = -1 - sqrt 6 / 2 ∧ 
    ∀ x : ℝ, 2 * x^2 + 4 * x - 1 = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end solve_quadratic_l372_372097


namespace min_value_of_M_l372_372132

noncomputable def a (n : ℕ) : ℝ := 3 + 2 * (n - 1)
noncomputable def b (n : ℕ) : ℝ := 2 ^ (n - 1)
noncomputable def a_over_b (n : ℕ) : ℝ := a n / b n

noncomputable def T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a_over_b (i + 1))

theorem min_value_of_M (M : ℝ) (hM : ∀ n : ℕ, n > 0 → T n < M) :
  10 ≤ M :=
sorry

end min_value_of_M_l372_372132


namespace dividend_is_5336_l372_372387

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) :
  (D * Q + R) = 5336 :=
by {
  sorry
}

end dividend_is_5336_l372_372387


namespace correct_number_of_ways_l372_372580

open Nat

def is_double (card : ℕ × ℕ) : Prop :=
  card.fst = card.snd

def num_cards : ℕ := 20 * 20

def valid_selections (cards : Finset (ℕ × ℕ)) : ℕ :=
  let doubles := cards.filter is_double
  let non_doubles := cards.filter (λ c => ¬ (is_double c))
  -- Count combinations of two doubles
  let case1 := Finset.card (Finset.pairwise doubles (•))
  -- Count combinations of one double and one non-double 
  let case2 := doubles.card * ((num_cards - doubles.card) - 19 - 19)
  case1 + case2

theorem correct_number_of_ways :
  valid_selections (Finset.univ : Finset (ℕ × ℕ)) = 7030 :=
by
  sorry

end correct_number_of_ways_l372_372580


namespace other_student_seat_number_l372_372207

-- Define the total number of students.
def total_students : ℕ := 48

-- Define the sample size.
def sample_size : ℕ := 4

-- Define the seat numbers in the sample.
def seat_numbers : set ℕ := {6, 30, 42}

-- Define the systematic sampling interval.
def sampling_interval : ℕ := total_students / sample_size

-- Define the theorem to prove the seat number of the other student in the sample.
theorem other_student_seat_number :
  ∃ (n : ℕ), n ∈ {x : ℕ | x < total_students ∧ ∀ y ∈ seat_numbers, (x = y) ∨ (x = y + sampling_interval ∨ x = y - sampling_interval)} ∧ n = 18 :=
sorry

end other_student_seat_number_l372_372207


namespace initial_fliers_l372_372912

theorem initial_fliers (F : ℕ) (morning_sent afternoon_sent remaining : ℕ) :
  morning_sent = F / 5 → 
  afternoon_sent = (F - morning_sent) / 4 → 
  remaining = F - morning_sent - afternoon_sent → 
  remaining = 1800 → 
  F = 3000 := 
by 
  sorry

end initial_fliers_l372_372912


namespace scatter_plot_line_properties_l372_372761

theorem scatter_plot_line_properties (points : List (ℝ × ℝ)) 
  (h : ∃ a b, ∀ p ∈ points, p.2 = a * p.1 + b)
  : (∀ p ∈ points, residual p = 0) ∧ (sum_sq_residuals points = 0) ∧ (correlation_coefficient points = 1) :=
begin
  sorry
end

end scatter_plot_line_properties_l372_372761


namespace students_attended_school_l372_372079

-- Definitions based on conditions
def total_students (S : ℕ) : Prop :=
  ∃ (L R : ℕ), 
    (L = S / 2) ∧ 
    (R = L / 4) ∧ 
    (5 = R / 5)

-- Theorem stating the problem
theorem students_attended_school (S : ℕ) : total_students S → S = 200 :=
by
  intro h
  sorry

end students_attended_school_l372_372079


namespace exists_zs_l372_372560

noncomputable def f (x : ℤ) : ℤ := x^8 + 4 * x^6 + 2 * x^4 + 28 * x^2 + 1

theorem exists_zs (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3)
    (z : ℤ) (hz : p ∣ f z) :
    ∃ z1 z2 z3 z4 z5 z6 z7 z8 : ℤ, 
      ∀ x : ℤ, ∃ k : ℤ, (f x) - (x - z1) * (x - z2) * (x - z3) * (x - z4) * (x - z5) * (x - z6) * (x - z7) * (x - z8) = p * k := 
sorry

end exists_zs_l372_372560


namespace angles_DEC_EQ_90_and_EDC_EQ_60_l372_372227

noncomputable def equilateral_triangle (A B C : Type) [euclidean : EuclideanGeometry A B C] :=
∀ (a b c : A), distance a b = distance b c ∧ distance b c = distance c a ∧ distance c a = distance a b

variables {A B C M P D E : Type} [euclidean : EuclideanGeometry A B C M P D E]

axiom parallel (x y z w : A) : Prop
axiom midpoint (u v m : A) : Prop
axiom center_of_equilateral (u v w d : A) : Prop

axiom equilateral_ABC : equilateral_triangle A B C
axiom line_parallel_AC (x : A) : parallel A B AC x
axiom meets_AB_M (a : A) : meets A B a
axiom meets_BC_P (b : A) : meets B C b
axiom center_D : center_of_equilateral B M P D
axiom midpoint_E : midpoint A P E

theorem angles_DEC_EQ_90_and_EDC_EQ_60 :
  angle D E C = 90 ∧ angle E D C = 60 :=
by
    sorry

end angles_DEC_EQ_90_and_EDC_EQ_60_l372_372227


namespace area_excircle_gteq_four_times_area_l372_372866

-- Define the area function
def area (A B C : Point) : ℝ := sorry -- Area of triangle ABC (this will be implemented later)

-- Define the centers of the excircles (this needs precise definitions and setup)
def excircle_center (A B C : Point) : Point := sorry -- Centers of the excircles of triangle ABC (implementation would follow)

-- Define the area of the triangle formed by the excircle centers
def excircle_area (A B C : Point) : ℝ :=
  let O1 := excircle_center A B C
  let O2 := excircle_center B C A
  let O3 := excircle_center C A B
  area O1 O2 O3

-- Prove the main statement
theorem area_excircle_gteq_four_times_area (A B C : Point) :
  excircle_area A B C ≥ 4 * area A B C :=
by sorry

end area_excircle_gteq_four_times_area_l372_372866


namespace square_from_diagonal_l372_372459

-- Define the vertices of the rectangle.
def A := (0, 0 : ℝ × ℝ)
def B := (0, 3 : ℝ × ℝ)
def C := (2, 3 : ℝ × ℝ)
def D := (2, 0 : ℝ × ℝ)

-- Calculate the diagonal of the rectangle.
def diagonal_length := Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2)

-- Prove that there exists a square with the side length equal to the diagonal of the rectangle.
theorem square_from_diagonal : ∃ (E F G : ℝ × ℝ), 
  dist A E = diagonal_length ∧
  dist E F = diagonal_length ∧
  dist F G = diagonal_length ∧
  dist G A = diagonal_length ∧
  ∠ A E F = 90 ∧
  ∠ E F G = 90 ∧
  ∠ F G A = 90 ∧
  ∠ G A E = 90 :=
sorry

end square_from_diagonal_l372_372459


namespace ram_krish_task_completion_l372_372086

theorem ram_krish_task_completion
  (ram_days : ℝ)
  (krish_efficiency_factor : ℝ)
  (task_time : ℝ) 
  (H1 : krish_efficiency_factor = 2)
  (H2 : ram_days = 27) 
  (H3 : task_time = 9) :
  (1 / task_time) = (1 / ram_days + 1 / (ram_days / krish_efficiency_factor)) := 
sorry

end ram_krish_task_completion_l372_372086


namespace first_group_correct_l372_372205

/-- Define the total members in the choir --/
def total_members : ℕ := 70

/-- Define members in the second group --/
def second_group_members : ℕ := 30

/-- Define members in the third group --/
def third_group_members : ℕ := 15

/-- Define the number of members in the first group by subtracting second and third groups members from total members --/
def first_group_members : ℕ := total_members - (second_group_members + third_group_members)

/-- Prove that the first group has 25 members --/
theorem first_group_correct : first_group_members = 25 := by
  -- insert the proof steps here
  sorry

end first_group_correct_l372_372205


namespace AB_difference_l372_372351

noncomputable def f (a m x : ℝ) : ℝ := x^2 - 2 * (a + m) * x + a^2
noncomputable def g (a m x : ℝ) : ℝ := -x^2 + 2 * (a - m) * x - a^2 + 2 * m^2

def H1 (a m x : ℝ) : ℝ := max (f a m x) (g a m x)
def H2 (a m x : ℝ) : ℝ := min (f a m x) (g a m x)

noncomputable def A (a m : ℝ) : ℝ := H1 a m (a + m)
noncomputable def B (a m : ℝ) : ℝ := H2 a m (a - m)

theorem AB_difference (a m : ℝ) : A a m - B a m = -4 * m^2 :=
by sorry

end AB_difference_l372_372351


namespace length_of_chord_l372_372721

-- Define the parabola E (y^2 = 2px) and specify p > 0
def parabola (p : ℝ) := { x y : ℝ // y^2 = 2 * p * x }

-- Define the circle F (x^2 + y^2 - 2x + 4y - 4 = 0)
def circle_F := { x y : ℝ // x^2 + y^2 - 2*x + 4*y - 4 = 0 }

-- Define the center of the circle
def center_of_circle_F := (1 : ℝ, -2 : ℝ)

-- Define the condition that the parabola passes through the center of the circle
def parabola_passes_through_center (p : ℝ) : Prop :=
  ∃ (x y : ℝ), parabola p x y ∧ (x, y) = center_of_circle_F

-- Define the statement to be proved
theorem length_of_chord (p : ℝ) (h : parabola_passes_through_center p) : 
  length_of_intersection_chord parabola.circleF parabola.directrix = 2 * sqrt 5 :=
sorry

end length_of_chord_l372_372721


namespace simplify_expression_l372_372094

/-!
# Simplification of the given expression
-/

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) :
  (8 / (5 * y ^ (-4)) * (5 * y ^ 3 / 4)) / y ^ 2 = 2 * y ^ 5 :=
by
  have h1 : 8 / (5 * y ^ (-4)) = 8 * y ^ 4 / 5, sorry,
  have h2 : (8 * y ^ 4 / 5) * (5 * y ^ 3 / 4) = 2 * y ^ 7, sorry,
  have h3 : (2 * y ^ 7) / y ^ 2 = 2 * y ^ 5, sorry,
  -- use the above steps to conclude
  rw [h1, h2, h3],

end simplify_expression_l372_372094


namespace men_count_l372_372891

/-- There are 80 passengers on the airplane, with 20 of them being children, and the number of men 
    and women is equal. Prove that there are 30 men on the airplane. -/
theorem men_count (total_passengers children : ℕ) (h_total : total_passengers = 80) 
  (h_children : children = 20) (h_equal : (total_passengers - children) % 2 = 0) : 
  let adults := total_passengers - children in let men := adults / 2 in men = 30 :=
by
  -- Definitions for the variables
  let adults := total_passengers - children
  have h_adults : adults = 60 := by
    rw [h_total, h_children]
    norm_num
  let men := adults / 2
  have h_men : men = 30 := by
    rw [h_adults]
    norm_num
  exact h_men

end men_count_l372_372891


namespace exists_root_in_interval_l372_372632

def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_root_in_interval :
  (f 2 < 0) → (f 3 > 0) → ∃ x0 : ℝ, 2 < x0 ∧ x0 < 3 ∧ f x0 = 0 :=
by sorry

end exists_root_in_interval_l372_372632


namespace binomial_sum_real_part_l372_372983

theorem binomial_sum_real_part :
  (1 / 2 ^ 1988) * (∑ n in Finset.range 995, (-3) ^ n * Nat.choose 1988 (2 * n)) = -1 / 2 :=
by
  sorry

end binomial_sum_real_part_l372_372983


namespace unique_solution_l372_372631

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (f x)) + f (f y) = f y + x

-- Define the main theorem
theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y, functional_eq f x y) → (∀ x, f x = x) :=
by
  intros h x
  -- Proof steps would go here
  sorry

end unique_solution_l372_372631


namespace product_of_roots_eq_negative_forty_nine_l372_372658

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372658


namespace complex_magnitude_example_l372_372740

theorem complex_magnitude_example (z : ℂ) (h : z = 1 + Complex.i) : Complex.abs (z^2 - 2*z) = 2 := 
by 
  rw [h]
  sorry

end complex_magnitude_example_l372_372740


namespace math_problem_l372_372975

theorem math_problem : ((-7)^3 / 7^2 - 2^5 + 4^3 - 8) = 81 :=
by
  sorry

end math_problem_l372_372975


namespace y_greater_than_one_l372_372122

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l372_372122


namespace length_of_platform_l372_372595

theorem length_of_platform (speed_kmh : ℝ) (time_pole : ℝ) (time_platform : ℝ) : 
    speed_kmh = 36 →
    time_pole = 14 → 
    time_platform = 51 → 
    ∃ (L : ℝ), L = 370 :=
by
  intros h1 h2 h3
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_train := speed_ms * time_pole
  let total_distance := speed_ms * time_platform
  let L := total_distance - length_train
  use L
  simp [h1, h2, h3]
  sorry -- This is where the proof would go, but is omitted

end length_of_platform_l372_372595


namespace tiffany_mile_fraction_l372_372242

/-- Tiffany's daily running fraction (x) for Wednesday, Thursday, and Friday must be 1/3
    such that both Billy and Tiffany run the same total miles over a week. --/
theorem tiffany_mile_fraction :
  ∃ x : ℚ, (3 * 1 + 1) = 1 + (3 * 2 + 3 * x) → x = 1 / 3 :=
by
  sorry

end tiffany_mile_fraction_l372_372242


namespace compound_interest_l372_372930

theorem compound_interest :
  ∃ P : ℝ, P = 740 / (1 + 0.05)^2 ∧ P ≈ 670.68 :=
begin
  use 740 / (1 + 0.05)^2,
  split,
  { -- Proof that P = 740 / (1 + 0.05)^2
    sorry
  },
  { -- Proof that the calculated P is approximately Rs. 670.68
    sorry
  }
end

end compound_interest_l372_372930


namespace last_episode_broadcast_date_l372_372571

theorem last_episode_broadcast_date :
  ∃ d, d = date.mk 2016 5 29 ∧ d.toMondayBasedWeekday = weekday.sunday :=
sorry

end last_episode_broadcast_date_l372_372571


namespace determine_n_l372_372433

theorem determine_n (n a b : ℤ) (h1 : n = a^2 + b^2) (h2 : Nat.gcd (Int.natAbs a) (Int.natAbs b) = 1)
    (h3 : ∀ p : ℕ, p.Prime → (p : ℤ) ≤ Int.sqrt n → (p ∣ a * b)) : n = 5 ∨ n = 13 :=
sorry

end determine_n_l372_372433


namespace problem_solution_l372_372760

noncomputable def sum_even_nums (n m : ℕ) : ℕ :=
  (finset.range (m / 2)).sum (λ k, 2 * (k + 1))

noncomputable def sum_odd_nums (n m : ℕ) : ℕ :=
  (finset.range ((m + 1) / 2)).sum (λ k, 2 * k + 1)

noncomputable def sum_primes (n : ℕ) : ℕ :=
  (finset.filter nat.prime (finset.range n)).sum id

theorem problem_solution : 
  let a := sum_even_nums 2 120,
      b := sum_odd_nums 1 119,
      c := sum_primes 120 in
  a - b + c = 1355 :=
by
  let a := sum_even_nums 2 120
  let b := sum_odd_nums 1 119
  let c := sum_primes 120
  sorry

end problem_solution_l372_372760


namespace three_colorable_l372_372784

/-
In a mathematical competition, some competitors are friends; friendship is mutual, that is, when A is a friend of B, then B is also a friend of A. 
We say that  n ≥ 3  different competitors A_1, A_2, ..., A_n form a weakly-friendly cycle if A_i is not a friend of A_{i+1}  for 1 ≤ i ≤ n (where A_{n+1} = A_1), and there are no other pairs of non-friends among the components of the cycle.

The following property is satisfied:
"for every competitor C and every weakly-friendly cycle S of competitors not including C, the set of competitors D in S which are not friends of C has at most one element"

Prove that all competitors of this mathematical competition can be arranged into three rooms, such that every two competitors in the same room are friends.
-/

def mutual_friendship (G : Type) [graph G] := ∀ (a b : G), (a ≠ b) → (a ~ b → b ~ a)

def weakly_friendly_cycle (G : Type) [graph G] (C : set G) :=
  ∃ (n : ℕ) (A : fin n → G), n ≥ 3 ∧ (∀ i, 1 ≤ i → i ≤ n → ¬ (A i ~ A (i + 1)) ∧ 
    (A n = A 0)) ∧ (∀ i ≠ j, ¬ (A i ~ A j))

def given_property (G : Type) [graph G] := 
  ∀ (C : G) (S : set G), weakly_friendly_cycle S ∧ C ∉ S → 
  (∃! D ∈ S, ¬ (C ~ D))

theorem three_colorable (G : Type) [graph G]
  (h1 : mutual_friendship G)
  (h2 : ∀ S, weakly_friendly_cycle S → given_property S) :
  ∃ (colors : G → fin 3), ∀ (a b : G), (a ~ b) → (colors a ≠ colors b) := sorry

end three_colorable_l372_372784


namespace find_base_a_l372_372346

theorem find_base_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (if a < 1 then a + a^2 else a^2 + a) = 12) : a = 3 := 
sorry

end find_base_a_l372_372346


namespace sum_of_m_l372_372411

theorem sum_of_m (A B C : Point) (P : ℕ → Point) (hAB : distance A B = 2) (hAC : distance A C = 2)
  (hPointsOnBC : ∀ i, 1 ≤ i ∧ i ≤ 100 → P i ∈ LineSegment B C) :
  ∑ i in finset.range (100), ((distance A (P i))^2 + (distance B (P i)) * (distance (P i) C)) = 400 :=
sorry

end sum_of_m_l372_372411


namespace chord_length_l372_372603

theorem chord_length (R : ℝ) (α : ℝ) (a : ℝ) 
    (h1 : 0 < α) (h2 : α < π) 
    (h_radius : 0 < R)
    (h3 : a = 2 * R * sin(α)) :
    a = 2 * R * sin(α) :=
sorry

end chord_length_l372_372603


namespace find_m_l372_372876

def h (x m : ℝ) := x^2 - 3 * x + m
def k (x m : ℝ) := x^2 - 3 * x + 5 * m

theorem find_m (m : ℝ) (h_def : ∀ x, h x m = x^2 - 3 * x + m) (k_def : ∀ x, k x m = x^2 - 3 * x + 5 * m) (key_eq : 3 * h 5 m = 2 * k 5 m) :
  m = 10 / 7 :=
by
  sorry

end find_m_l372_372876


namespace median_of_sequence_106_l372_372402

open Nat List

def sequence_up_to (n : ℕ) : List ℕ :=
  (range (n + 1)).bind (λ k => repeat k k)

noncomputable def median_of_list (l : List ℕ) : ℕ :=
  let sorted_l := List.qsort (≤) l
  let len := length sorted_l
  sorted_l[ (len - 1) / 2]

theorem median_of_sequence_106 :
  median_of_list (sequence_up_to 150) = 106 :=
by
  -- Proof omitted
  sorry

end median_of_sequence_106_l372_372402


namespace mod_product_l372_372859

theorem mod_product :
  (105 * 86 * 97) % 25 = 10 :=
by
  sorry

end mod_product_l372_372859


namespace day_of_week_2005_squared_l372_372527

theorem day_of_week_2005_squared :
  let today := "Sunday" in
  (2005^2) % 7 = 6 →
  let day_after_2005_squared := "Saturday" in
  day_after_2005_squared = "Saturday" := 
sorry

end day_of_week_2005_squared_l372_372527


namespace robins_initial_hair_length_l372_372842

variable (L : ℕ)

def initial_length_after_cutting := L - 11
def length_after_growth := initial_length_after_cutting L + 12
def final_length := 17

theorem robins_initial_hair_length : length_after_growth L = final_length → L = 16 := 
by sorry

end robins_initial_hair_length_l372_372842


namespace find_a_l372_372052

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372052


namespace smallest_number_of_eggs_proof_l372_372175

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l372_372175


namespace product_leq_one_half_l372_372796

variable {x : Fin 6 → ℝ}

theorem product_leq_one_half (h1 : ∑ i, x i ^ 2 = 6) (h2 : ∑ i, x i = 0) : (∏ i, x i) ≤ 1 / 2 :=
sorry

end product_leq_one_half_l372_372796


namespace joan_balloons_l372_372419

def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2
def total_balloons : ℕ := 16

theorem joan_balloons : sally_balloons + jessica_balloons = 7 ∧ total_balloons = 16 → total_balloons - (sally_balloons + jessica_balloons) = 9 :=
by
  sorry

end joan_balloons_l372_372419


namespace complex_sum_real_part_l372_372980

theorem complex_sum_real_part :
  (1 / 2 ^ 1988) * ∑ n in Finset.range 995, (-3 : ℂ) ^ n * (Nat.choose 1988 (2 * n) : ℂ) = -Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_sum_real_part_l372_372980


namespace smallest_positive_period_π_l372_372601

theorem smallest_positive_period_π :
  (∀ x, (cos (|2 * x|)) = (cos (|(2 * x + π)|))) ∧ 
  (∀ x, (|cos x|) = (|(cos (x + π))|)) ∧ 
  (∀ x, (cos (2 * x + π / 6)) = (cos ((2 * x + π / 6) + π))) ∧ 
  ¬(∀ x, (tan (2 * x - π / 4)) = (tan ((2 * x - π / 4) + π))) :=
by 
  sorry

end smallest_positive_period_π_l372_372601


namespace boat_speed_in_still_water_l372_372201

theorem boat_speed_in_still_water (x y : ℝ) :
  (80 / (x + y) + 48 / (x - y) = 9) ∧ 
  (64 / (x + y) + 96 / (x - y) = 12) → 
  x = 12 :=
by
  sorry

end boat_speed_in_still_water_l372_372201


namespace mark_and_carolyn_total_l372_372075

theorem mark_and_carolyn_total (m c : ℝ) (hm : m = 3 / 4) (hc : c = 3 / 10) :
    m + c = 1.05 :=
by
  sorry

end mark_and_carolyn_total_l372_372075


namespace credit_limit_l372_372831

theorem credit_limit (paid_tuesday : ℕ) (paid_thursday : ℕ) (remaining_payment : ℕ) (full_payment : ℕ) 
  (h1 : paid_tuesday = 15) 
  (h2 : paid_thursday = 23) 
  (h3 : remaining_payment = 62) 
  (h4 : full_payment = paid_tuesday + paid_thursday + remaining_payment) : 
  full_payment = 100 := 
by
  sorry

end credit_limit_l372_372831


namespace find_a_l372_372347
-- Import the required library to bring all necessary functionalities

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - (3 / 2) * x^2

-- Define the maximum condition
def max_cond (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≤ 1 / 6  -- Maximum value should not exceed 1/6

-- Define the interval condition
def interval_cond (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) → f a x ≥ 1 / 8  -- Function value should be >= 1/8 in the interval

-- The main theorem stating that under given conditions, a = 1
theorem find_a (a : ℝ) (max_cond a) (interval_cond a) : a = 1 :=
  sorry -- The proof is omitted


end find_a_l372_372347


namespace distance_from_start_farthest_trip_fuel_consumed_l372_372105

-- Define the list of mileages 
def mileages : List Int := [+15, -3, +14, -11, +10, -12, +4, -15, +16, -18]

-- Question 1
theorem distance_from_start : mileages.sum = 0 :=
  by sorry

-- Question 2
theorem farthest_trip : ∃ {k : ℕ}, 1 ≤ k ∧ k ≤ mileages.length ∧ ∀ {j : ℕ}, 1 ≤ j ∧ j ≤ mileages.length → (mileages.take k).sum ≥ (mileages.take j).sum :=
  by sorry

-- Question 3
theorem fuel_consumed : 4 * mileages.map Int.natAbs.sum = 472 :=
  by sorry

end distance_from_start_farthest_trip_fuel_consumed_l372_372105


namespace monotonic_decreasing_interval_l372_372508

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (3 < x) → monotonic_decreasing (λ x : ℝ, log (1/2) (abs (x - 3))) :=
by
  sorry

end monotonic_decreasing_interval_l372_372508


namespace factorial_expression_bound_l372_372281

theorem factorial_expression_bound (A B C D : ℕ) (h1 : A ≥ C) (h2 : B ≥ D) (h3 : 2011 = (A.factorial * C.factorial) / (B.factorial * D.factorial)) (h4 : ∀ A' B', (2011 = (A'.factorial * C.factorial) / (B'.factorial * D.factorial) → A' + B' ≥ A + B)) : |A - B| = 1 :=
sorry

end factorial_expression_bound_l372_372281


namespace ellipse_eccentricity_l372_372048

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372048


namespace Genevieve_coffee_drank_total_l372_372684

theorem Genevieve_coffee_drank_total :
  let small_thermos_ounces : ℝ := 250 * 0.0338
  let medium_thermos_ounces : ℝ := 400 * 0.0338
  let large_thermos_ounces : ℝ := 33.8
  small_thermos_ounces + medium_thermos_ounces + large_thermos_ounces = 55.77 :=
by
  let small_thermos_ounces : ℝ := 250 * 0.0338
  let medium_thermos_ounces : ℝ := 400 * 0.0338
  let large_thermos_ounces : ℝ := 33.8
  have h1 : small_thermos_ounces = 250 * 0.0338 := by rfl
  have h2 : medium_thermos_ounces = 400 * 0.0338 := by rfl
  have h3 : large_thermos_ounces = 33.8 := by rfl
  show small_thermos_ounces + medium_thermos_ounces + large_thermos_ounces = 55.77, from sorry

end Genevieve_coffee_drank_total_l372_372684


namespace ellipse_eccentricity_a_l372_372008

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372008


namespace backup_completion_time_l372_372573

def start_time := 22 + 0 / 60  -- 10:00 PM as 22:00 in 24-hour format
def half_time := 24 + 0.5  -- 12:30 AM as 24:30 in 24-hour format (next day)
def duration_half := half_time - start_time  -- Duration to complete half

def total_time := 2 * duration_half  -- Total time to complete the backup

def completion_time := start_time + total_time  -- Completion time

theorem backup_completion_time :
  completion_time = 27  -- 3:00 AM as 27:00 in 24-hour format (next day)
  := by
    unfold start_time
    unfold half_time
    unfold duration_half
    unfold total_time
    unfold completion_time
    norm_num
    sorry

end backup_completion_time_l372_372573


namespace problem_l372_372460

-- Declaring the geometric entities as pre-defined constants
variables (K L M N : ℝ) (A B C S P Q : ℝ)
variables (KL MN KN LM : ℝ)
variables (angleSAB : ℝ)

-- Declaring the conditions and required definitions

def is_regular_triangular_pyramid (A B C S : ℝ) : Prop := 
  Equilateral (A, B, C) ∧
  all_face_angles_equilateral (A, B, C, S) ∧
  all_edges_equal (A, B, C, S)

def conditions (K L M N : ℝ) (A B C S P Q : ℝ) : Prop :=
  is_regular_triangular_pyramid A B C S ∧
  (K.on_edge A C) ∧ (L.on_edge B C) ∧ (M.on_edge B S) ∧ (N.on_edge A S) ∧
  (KL = MN) ∧ (KL = 14) ∧ (MN = 14) ∧ (KN = LM) ∧ (KN = 25) ∧ (LM = 25) ∧
  (∃ Ω₁ Ω₂, inscribed_in_Ω₁ K L M N ∧ inscribed_in_Ω₂ K L M N ∧
    Ω₁.tangent_to KN MN LM ∧ Ω₂.tangent_to KL KN LM) ∧
  cone_in_pyramid (base Ω₁) S A B P ∧ cone_in_pyramid (base Ω₂) S C Q

def part_a (angleSAB : ℝ) : Prop := 
  angleSAB = real.arccos (3 / 5)

def part_b (CQ : ℝ) : Prop := 
  CQ = 77/5

theorem problem
  (h : conditions (K L M N) (A B C S P Q))
  : part_a angleSAB ∧ part_b (Q - C) := by
  sorry

end problem_l372_372460


namespace clarence_oranges_l372_372251

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l372_372251


namespace closest_perfect_square_to_320_l372_372546

theorem closest_perfect_square_to_320 : 
  ∃ n : ℕ, (n * n = 324) ∧ 
  ∀ m : ℕ, abs ((m * m) - 320) ≥ 4 :=
sorry

end closest_perfect_square_to_320_l372_372546


namespace product_of_solutions_product_of_all_t_l372_372674

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372674


namespace greatest_sum_of_products_l372_372605

def prime_faces := {2, 3, 5, 7, 11, 13}

def opposite_faces_sum_equal (p1 p2 p3 p4 p5 p6 : ℕ) : Prop :=
  p1 + p2 = p3 + p4 ∧ p1 + p2 = p5 + p6 ∧ p3 + p4 = p5 + p6

def eight_products_sum (p1 p2 p3 p4 p5 p6 : ℕ) : ℕ :=
  (p1 * p3 * p5 + p1 * p3 * p6 + p1 * p4 * p5 + p1 * p4 * p6 +
   p2 * p3 * p5 + p2 * p3 * p6 + p2 * p4 * p5 + p2 * p4 * p6)

theorem greatest_sum_of_products :
  ∃ (p1 p2 p3 p4 p5 p6 : ℕ), 
    {p1, p2, p3, p4, p5, p6} = prime_faces ∧
    opposite_faces_sum_equal p1 p2 p3 p4 p5 p6 ∧
    eight_products_sum p1 p2 p3 p4 p5 p6 = 3375 :=
by
  sorry

end greatest_sum_of_products_l372_372605


namespace reflections_on_circumcircle_of_orthocenter_l372_372444

open EuclideanGeometry

noncomputable def orthocenter (A B C : Point) : Point := sorry

def reflection (P Q R : Point) (H : Point) : Point := sorry

def circumcircle (A B C : Point) : Circle := sorry

theorem reflections_on_circumcircle_of_orthocenter
  (A B C : Point) (H : Point)
  (H_is_orthocenter : H = orthocenter A B C) :
  let H_A := reflection B C A H,
      H_B := reflection C A B H,
      H_C := reflection A B C H in
  H_A ∈ circumcircle A B C ∧
  H_B ∈ circumcircle A B C ∧
  H_C ∈ circumcircle A B C := 
sorry

end reflections_on_circumcircle_of_orthocenter_l372_372444


namespace cost_per_play_l372_372895

-- Conditions
def initial_money : ℝ := 3
def points_per_red_bucket : ℝ := 2
def points_per_green_bucket : ℝ := 3
def rings_per_play : ℕ := 5
def games_played : ℕ := 2
def red_buckets : ℕ := 4
def green_buckets : ℕ := 5
def total_games : ℕ := 3
def total_points : ℝ := 38

-- Point calculations
def points_from_red_buckets : ℝ := red_buckets * points_per_red_bucket
def points_from_green_buckets : ℝ := green_buckets * points_per_green_bucket
def current_points : ℝ := points_from_red_buckets + points_from_green_buckets
def points_needed : ℝ := total_points - current_points

-- Define the theorem statement
theorem cost_per_play :
  (initial_money / (games_played : ℝ)) = 1.50 :=
  sorry

end cost_per_play_l372_372895


namespace magnitude_a_minus_2b_l372_372323

-- Definition of unit vectors and mutual perpendicularity
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Defining conditions for a and b being unit vectors and mutually perpendicular
def unit_vector (v : V) := ∥v∥ = 1
def perpendicular (u v : V) := ⟪u, v⟫ = 0

-- Given conditions
axiom a_unit : unit_vector a
axiom b_unit : unit_vector b
axiom a_perpendicular_b : perpendicular a b

-- Theorem stating the desired property
theorem magnitude_a_minus_2b : ∥a - 2 • b∥ = real.sqrt 5 :=
by {
  -- Proof left as an exercise or to be filled in.
  sorry
}

end magnitude_a_minus_2b_l372_372323


namespace renata_donation_l372_372077

variable (D L : ℝ)

theorem renata_donation : ∃ D : ℝ, 
  (10 - D + 90 - L - 2 + 65 = 94) ↔ D = 4 :=
by
  sorry

end renata_donation_l372_372077


namespace even_indexed_theta_sum_l372_372630

-- Definitions and conditions from part a)
def complex_solution := 
  {z : ℂ // z^24 - z^16 - 1 = 0 ∧ complex.abs z = 1}

def theta_m (z : complex_solution) : ℝ := 
  complex.arg z.val

def sorted_thetas (theta_m_set : set ℝ) : list ℝ := 
  (theta_m_set.to_list.qsort (≤))

-- Generating the theorem statement
theorem even_indexed_theta_sum :
  let sols := {z : ℂ // z^24 - z^16 - 1 =0 ∧ complex.abs z = 1} in
  let theta_set := {θ | ∃ z : sols, θ = complex.arg z.val} in
  let sorted_list := sorted_thetas theta_set in
  let even_indices := sorted_list.enum.filter_map 
                        (λ (kθ : ℕ × ℝ), if kθ.1 % 2 = 1 then some kθ.2 else none) in
  even_indices.sum = 1620 :=
sorry

end even_indexed_theta_sum_l372_372630


namespace angle_x_is_36_l372_372397

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end angle_x_is_36_l372_372397


namespace find_prime_triples_l372_372644

theorem find_prime_triples :
  ∀ p x y : ℤ, p.prime ∧ x > 0 ∧ y > 0 ∧ p^x = y^3 + 1 →
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) := by
  sorry

end find_prime_triples_l372_372644


namespace tangent_line_equation_intersection_points_l372_372694

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (x + 1)^2
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g' (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem tangent_line_equation (x y x0 : ℝ) :
  function_vert_slope (g) x0 (-4, 0) = -Real.exp (-2) * (x + 4) := sorry

theorem intersection_points (a : ℝ) :
  (∃ x, (g x) = f x a) = 
  if a < 0 then 2 else 1 := sorry

end tangent_line_equation_intersection_points_l372_372694


namespace baking_powder_now_l372_372171

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_used : ℝ := 0.1

theorem baking_powder_now : 
  baking_powder_yesterday - baking_powder_used = 0.3 :=
by
  sorry

end baking_powder_now_l372_372171


namespace speed_of_man_is_5point5_l372_372213

def speed_of_man_in_still_water (downstream_distance upstream_distance downstream_time upstream_time : ℝ) : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let upstream_speed := upstream_distance / upstream_time
  (downstream_speed + upstream_speed) / 2

theorem speed_of_man_is_5point5 :
  speed_of_man_in_still_water 35 20 5 5 = 5.5 :=
by
  sorry

end speed_of_man_is_5point5_l372_372213


namespace frustum_height_l372_372526

noncomputable def frustum_volume (S S' h : ℝ) : ℝ := (1/3) * (S + real.sqrt (S * S') + S') * h

theorem frustum_height :
  ∀ (V : ℝ) (b1 b2 h : ℝ),
    V = 190000 → 
    b1 = 60 → 
    b2 = 40 → 
    let S := b1^2
    let S' := b2^2
    frustum_volume S S' h = V → 
    h = 75 :=
by {
  intros,
  sorry
}

end frustum_height_l372_372526


namespace no_positive_rational_solution_l372_372415

theorem no_positive_rational_solution :
  ¬ ∃ q : ℚ, 0 < q ∧ q^3 - 10 * q^2 + q - 2021 = 0 :=
by sorry

end no_positive_rational_solution_l372_372415


namespace product_of_values_t_squared_eq_49_l372_372647

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l372_372647


namespace proof_problem_l372_372703

def p := 8 + 7 = 16
def q := Real.pi > 3

theorem proof_problem :
  (¬p ∧ q) ∧ ((p ∨ q) = true) ∧ ((p ∧ q) = false) ∧ ((¬p) = true) := sorry

end proof_problem_l372_372703


namespace shortest_distance_to_tent_l372_372421

noncomputable def shortest_distance_juca_can_travel 
  (J B : Point)
  (r : Line)
  (C E : Point)
  (distance_between_CE: dist C E = 180) : ℝ :=
  180 * Real.sqrt 2

theorem shortest_distance_to_tent
  {J B : Point} 
  {r : Line} 
  {C E : Point}
  (h : dist (foot r J) (foot r B) = 180) : 
  shortest_distance_juca_can_travel J B r C E 180 = 180 * Real.sqrt 2 :=
sorry

end shortest_distance_to_tent_l372_372421


namespace volume_shaded_region_l372_372623

-- Defining the problem using the given conditions

def shadedRegion : Type :=
  {region // 
    ∃ v_strip h_strip, 
    (v_strip = 7 ∧ h_strip = 6 ∧ (v_strip * (h_strip - 1) + h_strip = 13))
  }

noncomputable def volume_of_solid (r : shadedRegion) : ℝ :=
  let v_strip_volume := π * (7:ℝ)^2 * 1
  let h_strip_volume := π * 1^2 * 5
  v_strip_volume + h_strip_volume

theorem volume_shaded_region (r : shadedRegion) : volume_of_solid r = 54 * π :=
  sorry

end volume_shaded_region_l372_372623


namespace sum_cubes_multiplied_by_25_cubed_l372_372539

theorem sum_cubes_multiplied_by_25_cubed :
    ((∑ i in (Finset.range 50).map (λ x, x + 1), i^3) + (∑ i in (Finset.range 50).map (λ x, x + 1), (-i)^3)) * 25^3 = 0 :=
by
  sorry

end sum_cubes_multiplied_by_25_cubed_l372_372539


namespace find_a_l372_372023

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372023


namespace find_x_l372_372362

open Real

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (2, -1)

noncomputable def x_solution := - (2 / 5)

theorem find_x (x : ℝ) (h1 : vec_a = (3, 4)) (h2 : vec_b = (2, -1)) (h3 : ∀ u v : ℝ × ℝ, u + v = (3 + 2*x, 4 - x) → dot_product (u + v) vec_b = 0) :
  x = x_solution := by
sorry

end find_x_l372_372362


namespace distance_difference_l372_372612

-- Define coordinates as vectors
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def O1 : ℝ × ℝ × ℝ := (0.5, 0.5, 1)
def B_mid : ℝ × ℝ × ℝ := (0.5, 0, 0)

-- Define the Euclidean distance between two 3D points
def euclidean_distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

-- Statement of the problem
theorem distance_difference :
  euclidean_distance O1 A - euclidean_distance O1 B_mid = (real.sqrt 6 - real.sqrt 5) / 2 :=
by
  sorry

end distance_difference_l372_372612


namespace roots_of_quadratic_eq_l372_372736

noncomputable def r : ℂ := sorry
noncomputable def s : ℂ := sorry

def roots_eq (h : 3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : Prop :=
  (1 / r^3) + (1 / s^3) = 1

theorem roots_of_quadratic_eq (h:3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : roots_eq h :=
sorry

end roots_of_quadratic_eq_l372_372736


namespace solve_for_a_l372_372000

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372000


namespace monotonic_intervals_when_a_zero_range_of_a_for_f_gt_sqrtx_l372_372345

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - a) / (log x)

theorem monotonic_intervals_when_a_zero :
  (∀ x, (0 < x ∧ x < exp 1 → deriv (fun x => x / log x) x < 0) ∧
       (x > exp 1 → deriv (fun x => x / log x) x > 0)) :=
begin
  sorry
end

theorem range_of_a_for_f_gt_sqrtx (a : ℝ) :
  (∀ x, 1 < x → (x - a) / (log x) > sqrt x) → (a ≤ 1) :=
begin
  sorry
end

end monotonic_intervals_when_a_zero_range_of_a_for_f_gt_sqrtx_l372_372345


namespace list_lengths_contradiction_l372_372072

theorem list_lengths_contradiction:
    let List_I := [3, 4, 8, 19] in
    let List_II_length := List_I.length + 1 in
    List_II_length - List_I.length = 6 → false :=
by
  intros
  sorry

end list_lengths_contradiction_l372_372072


namespace part1_part2_l372_372726

variable {x m : ℝ}

def a (x : ℝ) := (√3 * sin x, -1)
def b (x m : ℝ) := (cos x, m)

theorem part1 (h : m = √3) (h_parallel : a x = -1 / √3 * b x m) :
  (3 * sin x - cos x) / (sin x + cos x) = -3 := by
  sorry

def f (x m : ℝ) := 2 * ((a x).fst + (b x m).fst) * (b x m).snd + 2 * ((a x).snd + (b x m).snd) * (b x m).fst - 2 * m^2 - 1

theorem part2 (h_zero : ∃ x ∈ Icc 0 (π / 2), f x m = 0) :
  -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end part1_part2_l372_372726


namespace karen_tests_graded_l372_372422

theorem karen_tests_graded (n : ℕ) (T : ℕ) 
  (avg_score_70 : T = 70 * n)
  (combined_score_290 : T + 290 = 85 * (n + 2)) : 
  n = 8 := 
sorry

end karen_tests_graded_l372_372422


namespace complex_magnitude_example_l372_372741

theorem complex_magnitude_example (z : ℂ) (h : z = 1 + Complex.i) : Complex.abs (z^2 - 2*z) = 2 := 
by 
  rw [h]
  sorry

end complex_magnitude_example_l372_372741


namespace math_problem_l372_372230

-- Condition 1: f(x) is increasing when x > 0 and x < 0
def cond1 (f : ℝ → ℝ) := (∀ x > 0, f x > f (x - 1)) ∧ (∀ x < 0, f x < f (x - 1))

-- Condition 2: m = log_a 2, n = log_b 2, m > n implies a < b
def cond2 (a b : ℝ) := let m := Real.log 2 / Real.log a in
                       let n := Real.log 2 / Real.log b in
                       m > n → a < b

-- Condition 3: f(x) = x^2 + 2(a - 1)x + 2 is decreasing in (-∞, 4]
def cond3 (a : ℝ) := ∀ x ≤ 4, (x^2 + 2*(a - 1)*x + 2) * 2(x - 4) ≤ 0

-- Condition 4: y = log_(1/2)(x^2 + x - 2) is decreasing in (1, +∞)
def cond4 := ∀ x, (1 < x) → Real.log ((x^2 + x - 2) / 2) / Real.log (1/2) < 0

-- The equivalent proof problem statement
theorem math_problem : 
  (∀ f : ℝ → ℝ, cond1 f -> False) ∧ 
  (∀ a b : ℝ, cond2 a b) ∧
  (∀ a : ℝ, cond3 a -> a = -3) ∧
  (cond4) -> true :=
by sorry

end math_problem_l372_372230


namespace range_of_a_l372_372714

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then -x^2 - 2*a*x - 5 else a / x

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (is_increasing (f a) → a ∈ set.Icc (-2) (-1)) ∧
  (a ∈ set.Icc (-2) (-1) → is_increasing (f a)) :=
begin
  sorry
end

end range_of_a_l372_372714


namespace arithmetic_mean_of_two_digit_multiples_of_5_l372_372158

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5_l372_372158


namespace length_of_wall_correct_l372_372223

noncomputable def length_of_wall (s : ℝ) (w : ℝ) : ℝ :=
  let area_mirror := s * s
  let area_wall := 2 * area_mirror
  area_wall / w

theorem length_of_wall_correct : length_of_wall 18 32 = 20.25 :=
by
  -- This is the place for proof which is omitted deliberately
  sorry

end length_of_wall_correct_l372_372223


namespace gold_copper_alloy_ratio_l372_372183

theorem gold_copper_alloy_ratio 
  (water : ℝ) 
  (G : ℝ) 
  (C : ℝ) 
  (H1 : G = 10 * water)
  (H2 : C = 6 * water)
  (H3 : 10 * G + 6 * C = 8 * (G + C)) : 
  G / C = 1 :=
by
  sorry

end gold_copper_alloy_ratio_l372_372183


namespace cross_product_magnitude_l372_372263

variable (a b : ℝ^3)

-- Conditions
variable (h₁ : |a| = 2)
variable (h₂ : |b| = 5)
variable (h₃ : (a + b) ⬝ a = -2)  -- ⬝ denotes dot product

-- Goal (Question == Answer)
theorem cross_product_magnitude : |a × b| = 8 := by
  sorry

end cross_product_magnitude_l372_372263


namespace amoeba_population_at_11am_l372_372475

/-- Sarah observes an amoeba colony where initially there are 50 amoebas at 10:00 a.m. The population triples every 10 minutes and there are no deaths among the amoebas. Prove that the number of amoebas at 11:00 a.m. is 36450. -/
theorem amoeba_population_at_11am : 
  let initial_population := 50
  let growth_rate := 3
  let increments := 6  -- since 60 minutes / 10 minutes per increment = 6
  initial_population * (growth_rate ^ increments) = 36450 :=
by
  sorry

end amoeba_population_at_11am_l372_372475


namespace hyperbola_has_given_eccenctricity_l372_372353

noncomputable def hyperbola_eccentricity (a b x y : ℝ) : ℝ :=
  real.sqrt (1 + (b^2) / (a^2))

theorem hyperbola_has_given_eccenctricity
  (a b x y : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : (x = 3 ∧ (y = 2 * real.sqrt 6 ∨ y = -2 * real.sqrt 6)))
  (h4 : (x^2 / a^2 - y^2 / b^2 = 1)) 
  (h5 : y^2 = 8 * x) 
  (h6 : (2^2 + 0^2)^0.5 = 2) 
  : hyperbola_eccentricity a b x y = 2 :=
sorry

end hyperbola_has_given_eccenctricity_l372_372353


namespace largest_power_of_two_divides_express_l372_372161

theorem largest_power_of_two_divides_express (r : ℕ) : 
  r = 10^10 - 2^10 → (2^13 ∣ r ∧ ¬ 2^14 ∣ r) :=
begin
  sorry
end

end largest_power_of_two_divides_express_l372_372161


namespace plane_equation_exists_l372_372267

noncomputable def equation_of_plane (A B C D : ℤ) (hA : A > 0) (hGCD : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) : Prop :=
∃ (x y z : ℤ),
  x = 1 ∧ y = -2 ∧ z = 2 ∧ D = -18 ∧
  (2 * x + (-3) * y + 5 * z + D = 0) ∧  -- Point (2, -3, 5) satisfies equation
  (4 * x + (-3) * y + 6 * z + D = 0) ∧  -- Point (4, -3, 6) satisfies equation
  (6 * x + (-4) * y + 8 * z + D = 0)    -- Point (6, -4, 8) satisfies equation

theorem plane_equation_exists : equation_of_plane 1 (-2) 2 (-18) (by decide) (by decide) :=
by
  -- Proof is omitted
  sorry

end plane_equation_exists_l372_372267


namespace paths_H_to_J_via_I_l372_372487

def binom (n k : ℕ) : ℕ := Nat.choose n k

def paths_from_H_to_I : ℕ :=
  binom 7 2  -- Calculate the number of paths from H(0,7) to I(5,5)

def paths_from_I_to_J : ℕ :=
  binom 8 3  -- Calculate the number of paths from I(5,5) to J(8,0)

theorem paths_H_to_J_via_I : paths_from_H_to_I * paths_from_I_to_J = 1176 := by
  -- This theorem states that the number of paths from H to J through I is 1176
  sorry  -- Proof to be provided

end paths_H_to_J_via_I_l372_372487


namespace symmetric_about_y_axis_l372_372506

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  sorry

end symmetric_about_y_axis_l372_372506


namespace crank_slider_motion_speed_of_M_l372_372260

theorem crank_slider_motion (t : ℝ) : 
  let ω := 10, OA := 90 in
  (∀ t, (90 * (Real.cos (ω * t)), 90 * (Real.sin (ω * t)))) = (90 * (Real.cos (10 * t)), 90 * (Real.sin (10 * t))) :=
by
  sorry

theorem speed_of_M (t : ℝ) : 
  let ω := 10, OA := 90, AB := 90, AM := AB / 2 in
  ∀ t, sqrt ((- (OA * ω) * (Real.sin (ω * t)))^2 + (OA * ω * (Real.cos (ω * t)))^2) = 900 :=
by
  sorry

end crank_slider_motion_speed_of_M_l372_372260


namespace B_50_l372_372704

noncomputable def B (n : ℕ) (a : ℝ) : ℝ := ∑ i in finset.range n, (3 * (i+1) - 2) * a^(i+1)

theorem B_50 : 
  ∃ a : ℝ, (4 * a - 3 < 3 - 2 * a^2) ∧ y = f (2 * x - 3) ∧ 
  (∀ x : ℝ, y = f (-x + 3)) →
  a = -1 → B 50 a = 75 := 
sorry

end B_50_l372_372704


namespace letters_per_large_envelope_l372_372237

theorem letters_per_large_envelope
  (total_letters : ℕ)
  (small_envelope_letters : ℕ)
  (large_envelopes : ℕ)
  (large_envelopes_count : ℕ)
  (h1 : total_letters = 80)
  (h2 : small_envelope_letters = 20)
  (h3 : large_envelopes_count = 30)
  (h4 : total_letters - small_envelope_letters = large_envelopes)
  : large_envelopes / large_envelopes_count = 2 :=
by
  sorry

end letters_per_large_envelope_l372_372237


namespace min_value_exists_l372_372357

-- Given sequence an = 3/(2n - 11)
def a_n (n : ℕ) : ℝ := 3 / (2 * n - 11)

-- Sum of the first n terms Sn
def S_n (n : ℕ) : ℝ := (Finset.range n).sum (λ i, a_n (i + 1))

-- Theorem stating both an and Sn have a minimum value
theorem min_value_exists : ∃ m₁ m₂, (∃ n, a_n n = m₁) ∧ (∃ n, S_n n = m₂) :=
sorry

end min_value_exists_l372_372357


namespace triangle_area_k_values_l372_372518

theorem triangle_area_k_values (a b : ℝ) (area : ℝ) (k₁ k₂ : ℝ) :
  a = 3 → b = 5 → area = 6 → 
  (∃ k, k ∈ {k₁, k₂}) → k₁ ≠ k₂ →
  let x := k₁
  let y := k₂
  |x^2 - y^2| = 36 :=
by
  intros
  sorry

end triangle_area_k_values_l372_372518


namespace combined_percentage_score_l372_372454

-- Defining the conditions as per the problem statement
def correctness_on_first_test := (0.6 : ℝ) * 15 = 9
def correctness_on_second_test := (0.85 : ℝ) * 20 = 17
def correctness_on_third_test := (0.75 : ℝ) * 40 = 30

-- Using the above definitions to come up with the combined score
def total_correct_problems := 9 + 17 + 30 = 56
def total_problems := 15 + 20 + 40 = 75

-- Proving that the overall percentage score is 75%
theorem combined_percentage_score :
  correctness_on_first_test →
  correctness_on_second_test →
  correctness_on_third_test →
  total_correct_problems →
  total_problems →
  ((56 / 75) * 100) ≈ 75 :=
by
  sorry

end combined_percentage_score_l372_372454


namespace coefficient_x3_in_binomial_expansion_l372_372516

theorem coefficient_x3_in_binomial_expansion :
  (∑ i in Finset.range (7), (3 - 1) ^ 6) = 64 → 
  (∑ i in Finset.range (7), (-1)^i * Nat.choose 6 i * 3^(6-i) * (3 = 3)) = -540 :=
by 
  intro h_sum
  have h : (3 - 1 : ℝ) ^ 6 = 64 := by norm_num
  sorry

end coefficient_x3_in_binomial_expansion_l372_372516


namespace determinant_example_l372_372616

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l372_372616


namespace lines_intersection_l372_372579

section

variables {s v x y : ℝ}

def line1 (s : ℝ) : ℝ × ℝ := (5 + 3 * s, 1 - 2 * s)
def line2 (v : ℝ) : ℝ × ℝ := (2 + 5 * v, 8 - 3 * v)

theorem lines_intersection :
  ∃ s v : ℝ, line1 s = (-73, 53) ∧ line2 v = (-73, 53) :=
by
  use (-26)
  use (-15)
  constructor
  case_left
  simp [line1]
  case_right
  simp [line2]

end

end lines_intersection_l372_372579


namespace sum_possible_B_divisibility_l372_372270

theorem sum_possible_B_divisibility : 
  (∃ B : ℕ, (B < 10) ∧ (954312000 + B * 100 + 93) % 7 = 0 ∧ (954312000 + B * 100 + 93) % 5 = 0) → 
  ∑ (B : ℕ) in { b ∈ finset.range 10 | (954312000 + b * 100 + 93) % 7 = 0 }, b = 6 :=
by sorry

end sum_possible_B_divisibility_l372_372270


namespace AdvancedVowelSoup_l372_372599

noncomputable def AdvancedVowelSoup.sequence_count : ℕ :=
  let total_sequences := 7^7
  let vowel_only_sequences := 5^7
  let consonant_only_sequences := 2^7
  total_sequences - vowel_only_sequences - consonant_only_sequences

theorem AdvancedVowelSoup.valid_sequences : AdvancedVowelSoup.sequence_count = 745290 := by
  sorry

end AdvancedVowelSoup_l372_372599


namespace floor_of_sum_eq_l372_372442

theorem floor_of_sum_eq (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x^2 + y^2 = 2500) (hzw : z^2 + w^2 = 2500) (hxz : x * z = 1200) (hyw : y * w = 1200) :
  ⌊x + y + z + w⌋ = 140 := by
  sorry

end floor_of_sum_eq_l372_372442


namespace range_of_g_l372_372997

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - 3 * x) / (2 + 3 * x))

theorem range_of_g : 
  ∀ y ∈ set.range g, y = -3 * Real.pi / 4 ∨ y = Real.pi / 4 := 
by
  sorry

end range_of_g_l372_372997


namespace min_value_function_l372_372737

theorem min_value_function (x : ℝ) (h : x > 0) : 
  ∃ y, y = (x^2 + x + 25) / x ∧ y ≥ 11 :=
sorry

end min_value_function_l372_372737


namespace cubic_polynomial_Q_l372_372438

noncomputable def cubic_poly : Polynomial ℝ := Polynomial.Coeff (Polynomial.X^3 + 4 * Polynomial.X^2 + 6 * Polynomial.X + 8)

theorem cubic_polynomial_Q (p q r : ℝ)
  (hpqr : (Polynomial.roots cubic_poly).toFinset = {p, q, r})
  (Q : Polynomial ℝ)
  (hQp : Q.eval p = q + r)
  (hQq : Q.eval q = p + r)
  (hQr : Q.eval r = p + q)
  (hQsum : Q.eval (p + q + r) = -20) :
  Q = (frac 5 4) * (Polynomial.X^3 + 4 * Polynomial.X^2 + 6 * Polynomial.X + 8) 
      - Polynomial.X - 4 :=
sorry

end cubic_polynomial_Q_l372_372438


namespace smallest_n_to_make_y_perfect_square_l372_372956

/-- Given y = 2^4 * 3^3 * 5^4 * 7^2 * 2^7 * 3^7 * 2^9 * 3^{10}, 
    prove that the smallest positive integer n such that n * y is a perfect square is 1. -/
theorem smallest_n_to_make_y_perfect_square :
  ∃ n : ℕ, n = 1 ∧ (∃ m : ℕ, m^2 = n * (2^4 * 3^3 * 5^4 * 7^2 * 2^7 * 3^7 * 2^9 * 3^{10)) :=
by 
  sorry

end smallest_n_to_make_y_perfect_square_l372_372956


namespace solve_S20_minus_2S10_l372_372197

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → S n = (n / (n - 1 : ℝ)) * (a n ^ 2 - a 1 ^ 2))

theorem solve_S20_minus_2S10 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a S →
    S 20 - 2 * S 10 = 50 :=
by
  intros
  sorry

end solve_S20_minus_2S10_l372_372197


namespace Clarence_total_oranges_l372_372253

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l372_372253


namespace sum_binom_expr_eq_neg_half_l372_372987

def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else (n.factorial) / (k.factorial * (n - k).factorial)

noncomputable def sum_binom_expr : ℝ :=
  (1 / 2 ^ 1988) * ∑ n in Finset.range (995), (-3 : ℝ) ^ n * binom 1988 (2 * n)

theorem sum_binom_expr_eq_neg_half : sum_binom_expr = -1 / 2 := by
  sorry

end sum_binom_expr_eq_neg_half_l372_372987


namespace pascal_triangle_mult_3_l372_372989

theorem pascal_triangle_mult_3 :
  let rows := List.range 30
  List.count (λ n, ∀ k, 1 ≤ k ∧ k ≤ n - 1 → (Nat.choose n k) % 3 = 0) (rows.erase 0).erase 1 = 2 := by
  sorry

end pascal_triangle_mult_3_l372_372989


namespace concurrency_of_lines_l372_372330

-- Define the points and the circles in the Lean context
variables {S S1 S2 S3 : Type} [circle S] [circle S1] [circle S2] [circle S3]
variables {A B C A1 B1 C1 : Point}

-- Define the tangency and geometric conditions
variables (tangency1 : tangent S1 S A1)
variables (tangency2 : tangent S2 S B1)
variables (tangency3 : tangent S3 S C1)
variables (tangent_AB : tangent_to_triangle_side S1 A B)
variables (tangent_BC : tangent_to_triangle_side S2 B C)
variables (tangent_CA : tangent_to_triangle_side S3 C A)

-- Define the theorem statement
theorem concurrency_of_lines :
  concurrent (line_through_points A A1) (line_through_points B B1) (line_through_points C C1) :=
  sorry

end concurrency_of_lines_l372_372330


namespace fencing_problem_l372_372914

noncomputable def fencingRequired (L A W F : ℝ) := (A = L * W) → (F = 2 * W + L)

theorem fencing_problem :
  fencingRequired 25 880 35.2 95.4 :=
by
  sorry

end fencing_problem_l372_372914


namespace problem_sum_exponents_l372_372525

theorem problem_sum_exponents (s : ℕ) (b : Fin s → ℤ) (m : Fin s → ℕ)
  (h1 : ∀ i : Fin s, b i = 1 ∨ b i = -1)
  (h2 : Finset.univ.card = s)
  (h3 : ∀ i : Fin s, ∃! j : Fin s, m i > m j)
  (h4 : ∑ i in Finset.univ, b i * (3 ^ m i) = 2021) :
  ∑ i in Finset.univ, m i = 19 :=
sorry

end problem_sum_exponents_l372_372525


namespace range_of_a_l372_372966

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def is_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ ⦃x y⦄, x < y → f y < f x

def in_interval (a : ℝ) : Prop := 
a > 0 ∧ a < 2/3

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, x ∈ Ioo (-1 : ℝ) (1 : ℝ) → x ∈ dom f) 
  (h3 : is_decreasing_function f) 
  (h4 : f (1-a) + f (1-2a) < 0):
  in_interval a :=
sorry

end range_of_a_l372_372966


namespace displacement_increment_correct_l372_372872

-- Define the equation of motion
def motion_equation (t : ℝ) : ℝ := 2 * t^2

-- Define the displacement increment in terms of d
def displacement_increment (d : ℝ) : ℝ :=
  let s := motion_equation in
  s (2 + d) - s 2

-- The statement to prove
theorem displacement_increment_correct (d : ℝ) : displacement_increment d = 8 * d + 2 * d^2 :=
  sorry

end displacement_increment_correct_l372_372872


namespace find_a_l372_372060

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372060


namespace electric_field_and_potential_at_midpoint_l372_372971

theorem electric_field_and_potential_at_midpoint (q : ℝ) (a : ℝ) (k : ℝ) 
  (positive_charge_positions : list (ℝ × ℝ)) 
  (negative_charge_positions : list (ℝ × ℝ))
  (pos_charge_correctness : positive_charge_positions = [(0, 0), (a, 0), (0, a)]) 
  (neg_charge_correctness : negative_charge_positions = [(a, a)]) :
  let E := (3600 : ℝ × ℝ) -- The magnitude of the electric field
  let φ := 160 -- The electric potential
in sqrt((8 * k * q / a ^ 2) ^ 2 + (16 * k * q / (5 * sqrt(5) * a ^ 2)) ^ 2) ≈ 3600 * k * q / a ^ 2 
   ∧ 2 * (k * q / (sqrt(a ^ 2 + (a / 2) ^ 2))) = 160 :=
by
  sorry

end electric_field_and_potential_at_midpoint_l372_372971


namespace verify_ages_l372_372212

noncomputable def correct_ages (S M D W : ℝ) : Prop :=
  (M = S + 29) ∧
  (M + 2 = 2 * (S + 2)) ∧
  (D = S - 3.5) ∧
  (W = 1.5 * D) ∧
  (S = 27) ∧
  (M = 56) ∧
  (D = 23.5) ∧
  (W = 35.25)

theorem verify_ages : ∃ (S M D W : ℝ), correct_ages S M D W :=
by
  sorry

end verify_ages_l372_372212


namespace digits_1_left_of_2_and_3_l372_372871

open Nat

theorem digits_1_left_of_2_and_3 :
  let digits := {1, 2, 3, 4, 5, 6}
  let total_permutations := 6!
  let favorable_arrangements := 288
  ∀ x : Finset ℕ, x = digits → 
    let arrangements := x.to_list.permutations
    arrangements.length = total_permutations →
    (∃ favorable : Finset (List ℕ), 
      favorable.card = favorable_arrangements ∧ 
      ∀ lst ∈ favorable, 
        lst.index_of 1 < lst.index_of 2 ∧ lst.index_of 1 < lst.index_of 3) :=
  sorry

end digits_1_left_of_2_and_3_l372_372871


namespace find_sixth_term_l372_372503

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l372_372503


namespace jason_work_experience_l372_372802

theorem jason_work_experience :
  let months_in_year := 12 in
  let bartending_years := 9 in
  let managing_years := 3 in
  let managing_additional_months := 6 in
  (bartending_years * months_in_year) + (managing_years * months_in_year + managing_additional_months) = 150 := sorry

end jason_work_experience_l372_372802


namespace james_savings_l372_372799

theorem james_savings (weekly_allowance : ℕ) (weeks : ℕ) (video_game_fraction : ℕ) (book_fraction : ℕ) :
  weekly_allowance = 10 →
  weeks = 4 →
  video_game_fraction = 2 →
  book_fraction = 4 →
  let total_savings := weekly_allowance * weeks in
  let after_video_game := total_savings / video_game_fraction in
  let remaining_after_video_game := total_savings - after_video_game in
  let after_book := remaining_after_video_game / book_fraction in
  let final_amount := remaining_after_video_game - after_book in
  final_amount = 15 :=
by
  intros hw ha hv hb
  rw [hw, ha, hv, hb]
  let total_savings := weekly_allowance * weeks 
  let after_video_game := total_savings / video_game_fraction
  let remaining_after_video_game := total_savings - after_video_game
  let after_book := remaining_after_video_game / book_fraction
  let final_amount := remaining_after_video_game - after_book
  have h1 : total_savings = 40 := by
    rw [hw, ha]
  have h2 : after_video_game = 20 := by
    rw [hv, h1]
  have h3 : remaining_after_video_game = 20 := by
    exact sub_eq_of_eq_add' (by rw [hv, h1])
  have h4 : after_book = 5 := by
    rw [hb, h3]
  have h5 : final_amount = 15 := by
    exact sub_eq_of_eq_add (by rw [hb, h3])
  exact h5

end james_savings_l372_372799


namespace arithmetic_mean_of_two_digit_multiples_of_5_l372_372157

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5_l372_372157


namespace value_of_x_l372_372135

def x (y : ℝ) : ℝ := 1 / 2 * y
def y (z : ℝ) : ℝ := 1 / 4 * z
def z : ℝ := 100

theorem value_of_x : x (y z) = 12.5 := 
by
  sorry

end value_of_x_l372_372135


namespace find_a_l372_372057

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372057


namespace steven_set_aside_pears_l372_372100

theorem steven_set_aside_pears :
  ∀ (apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape : ℕ),
    apples = 4 →
    grapes = 9 →
    neededSeeds = 60 →
    seedPerApple = 6 →
    seedPerPear = 2 →
    seedPerGrape = 3 →
    (neededSeeds - 3) = (apples * seedPerApple + grapes * seedPerGrape + pears * seedPerPear) →
    pears = 3 :=
by
  intros apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape
  intros h_apple h_grape h_needed h_seedApple h_seedPear h_seedGrape
  intros h_totalSeeds
  sorry

end steven_set_aside_pears_l372_372100


namespace inequality_must_hold_l372_372170

theorem inequality_must_hold (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_must_hold_l372_372170


namespace determine_quartic_polynomial_l372_372857

noncomputable def q (x : ℝ) : ℝ := x^4 + x^3 - 19*x^2 - 116*x + 120

theorem determine_quartic_polynomial (h_monic : ∀ x: ℝ, polynomial.degree q = 4 ∧ q.monic) 
  (h_real : ∀ x: ℝ, q.coefficients_in ℝ) 
  (h_root : q (2 + ⟨0, sorry⟩) = 0)
  (h_q0 : q 0 = -120) : 
  q = λ x, x^4 + x^3 - 19*x^2 - 116*x + 120 :=
by
  sorry

end determine_quartic_polynomial_l372_372857


namespace final_sum_is_2S_l372_372886
noncomputable def sum_of_final_two_numbers (a b S : ℕ) (h : a + b = S) : ℕ :=
  2 * (a + 5) + 2 * (b - 5)

theorem final_sum_is_2S (a b S : ℕ) (h : a + b = S) : sum_of_final_two_numbers a b S h = 2 * S :=
begin
  sorry
end

end final_sum_is_2S_l372_372886


namespace distinct_sums_count_l372_372789

theorem distinct_sums_count : 
  let s : Finset (ℕ) := 
    { (1 * 2) + (3 * 4), 
      (1 * 3) + (2 * 4), 
      (1 * 4) + (2 * 3) }
  in s.card = 3 :=
by
  let s : Finset (ℕ) := 
    { (1 * 2) + (3 * 4), 
      (1 * 3) + (2 * 4), 
      (1 * 4) + (2 * 3) }
  exact Finset.card_eq.mpr (eq.refl 3)

end distinct_sums_count_l372_372789


namespace vanya_can_create_symmetric_figure_l372_372078

-- Definition of the initial figure F having no axes of symmetry
constant F : Type
constant no_axes_of_symmetry : F → Prop

-- Definition of the resulting figure F' with four axes of symmetry
constant F' : Type
constant four_axes_of_symmetry : F' → Prop

-- One shading operation that transforms F into F'
constant shade_one_cell : F → F'

-- Hypothesis: The starting figure F has no axes of symmetry
axiom H1 : ∃ f : F, no_axes_of_symmetry f

-- Conclusion: After shading one cell, the resulting figure F' could have four axes of symmetry
theorem vanya_can_create_symmetric_figure :
  ∀ f : F, no_axes_of_symmetry f → ∃ f' : F', four_axes_of_symmetry f' :=
sorry

end vanya_can_create_symmetric_figure_l372_372078


namespace angle_ZXY_eq_20_l372_372864

noncomputable def equilateral_triangle (A B C : Point) : Prop :=
  ∀ (a b c : ℝ), (side_length A B = a) ∧ (side_length B C = b) ∧ (side_length A C = c) ∧ a = b ∧ b = c

noncomputable def semicircle_center (O : Point) (A B : Point) : Prop :=
  ∃ r : ℝ, (distance O A = r) ∧ (distance O B = r) ∧ ∀ (P : Point), (distance O P = r) → (P = A ∨ P = B)

noncomputable def on_semicircle (X O : Point) (r : ℝ) : Prop :=
  distance O X = r

noncomputable def isosceles_triangle (X Y Z : Point) : Prop :=
  (side_length X Y = side_length X Z)

noncomputable def vertically_opposite_angles (APZ OPX : Angle) : Prop :=
  APZ = OPX

theorem angle_ZXY_eq_20
  {A B C P O X Y Z : Point}
  (h1 : equilateral_triangle A B C)
  (h2 : semicircle_center O A B)
  (h3 : on_semicircle X O (side_length O A))
  (h4 : meets_AC_at X Y Z A C)
  (h5 : isosceles_triangle X Y Z) :
  angle ZXY = 20° := sorry

end angle_ZXY_eq_20_l372_372864


namespace sales_fifth_month_l372_372576

-- Definitions based on conditions
def sales1 : ℝ := 5420
def sales2 : ℝ := 5660
def sales3 : ℝ := 6200
def sales4 : ℝ := 6350
def sales6 : ℝ := 8270
def average_sale : ℝ := 6400

-- Lean proof problem statement
theorem sales_fifth_month :
  sales1 + sales2 + sales3 + sales4 + sales6 + s = 6 * average_sale  →
  s = 6500 :=
by
  sorry

end sales_fifth_month_l372_372576


namespace max_dot_product_l372_372409

theorem max_dot_product
  (A B C : ℝ)
  (AC : ℝ)
  (h1 : 2 * B = A + C)
  (h2 : AC = 2)
  (h3 : A + B + C = real.pi) :
  ∃ max_val : ℝ, max_val = 2 + (4 * real.sqrt 3) / 3 ∧
  ∃ angles : ℝ × ℝ, angles = (A, C) ∧
  ∃ vector : ℝ, ∃ B_vector : ℝ, B_vector = B ∧
  let (a, b, c) := (A, B, C) in 
  let dot_product := sorry in -- Place calculation of dot product here based on conditions
  dot_product = max_val := 
sorry

end max_dot_product_l372_372409


namespace sum_of_coefficients_l372_372271

theorem sum_of_coefficients:
  (x^3 + 2*x + 1) * (3*x^2 + 4) = 28 :=
by
  sorry

end sum_of_coefficients_l372_372271


namespace probability_four_dice_equal_sum_l372_372304

noncomputable def fair_dice := finset.range 1 7
noncomputable def total_outcomes := 6^4

def probability_same_sum (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 ∈ fair_dice ∧ d2 ∈ fair_dice ∧ d3 ∈ fair_dice ∧ d4 ∈ fair_dice ∧
  (d1 + d2 = d3 + d4)

theorem probability_four_dice_equal_sum :
  ∃ (favorable_outcomes : ℕ), (6 * favorable_outcomes = 900) → (favorable_outcomes / total_outcomes = 25/36) :=
by
  use 150
  intros h
  rw total_outcomes
  linarith

end probability_four_dice_equal_sum_l372_372304


namespace five_wednesdays_implies_five_saturdays_in_august_l372_372858

theorem five_wednesdays_implies_five_saturdays_in_august (N : ℕ) (H1 : ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ w ∈ ws, w < 32 ∧ (w % 7 = 3)) (H2 : July_days = 31) (H3 : August_days = 31):
  ∀ w : ℕ, w < 7 → ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ sat ∈ ws, (sat % 7 = 6) :=
by
  sorry

end five_wednesdays_implies_five_saturdays_in_august_l372_372858


namespace algebraic_sum_of_tangent_lengths_is_zero_l372_372922

theorem algebraic_sum_of_tangent_lengths_is_zero
  (A : Point) (circle : Circle) (points : List Point)
  (hA_not_in_circle : ¬ A ∈ circle)
  (hClosedPath : points.head = A ∧ points.last = A)
  (hTangentsDefined : ∀ P, P ∈ points → is_tangent_segment_in_closed_path A circle P)
  (hSignConvention : ∀ (P Q : Point), 
    (is_closer_to_center P Q circle → sign (length_tangent P Q circle) = +1) ∧ 
    (is_away_from_center P Q circle → sign (length_tangent P Q circle) = -1)) :
  ∑ (i : Fin points.length), sign (length_tangent points[i] points[((i + 1) % points.length)] circle) * 
    length_tangent points[i] points[((i + 1) % points.length)] circle = 0 := by
  sorry

end algebraic_sum_of_tangent_lengths_is_zero_l372_372922


namespace find_m_l372_372734

theorem find_m (a0 a1 a2 a3 a4 a5 a6 m : ℝ) 
  (h1 : (1 + m) ^ 6 = a0 + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  m = 1 ∨ m = -3 := 
  sorry

end find_m_l372_372734


namespace Nicole_fish_tanks_l372_372457

-- Definition to express the conditions
def first_tank_water := 8 -- gallons
def second_tank_difference := 2 -- fewer gallons than first tanks
def num_first_tanks := 2
def num_second_tanks := 2
def total_water_four_weeks := 112 -- gallons
def weeks := 4

-- Calculate the total water per week
def water_per_week := (num_first_tanks * first_tank_water) + (num_second_tanks * (first_tank_water - second_tank_difference))

-- Calculate the total number of tanks
def total_tanks := num_first_tanks + num_second_tanks

-- Proof statement
theorem Nicole_fish_tanks : total_water_four_weeks / water_per_week = weeks → total_tanks = 4 := by
  -- Proof goes here
  sorry

end Nicole_fish_tanks_l372_372457


namespace population_2012_l372_372781

theorem population_2012 (P : ℕ → ℤ) (k : ℤ) : 
  P(2010) = 78 → 
  P(2011) = 127 → 
  P(2013) = 271 → 
  (∀ n, P(n+2) - P(n) = k * P(n+1)) →
  P(2012) = 180 :=
by
  intros h2010 h2011 h2013 h_prop
  sorry

end population_2012_l372_372781


namespace milk_needed_to_bake_8_cookies_l372_372897

-- Given constants and conditions
constant quarts_to_bake_24_cookies : ℕ := 5
constant cups_per_quart : ℕ := 4
constant cookies_per_batch : ℕ := 24
constant cookies_needed : ℕ := 8

-- Prove that the amount of milk needed to bake 8 cookies is (20/3) cups
theorem milk_needed_to_bake_8_cookies :
  let cups_to_bake_24_cookies := quarts_to_bake_24_cookies * cups_per_quart in
  cookies_needed * cups_to_bake_24_cookies = (cookies_per_batch / cookies_needed) * cups_per_quart := by
  let cups_to_bake_24_cookies := 5 * 4 in
  let proportion := 24 / 8 in
  let needed_cups := cups_to_bake_24_cookies / proportion in
  have : needed_cups = 20 / 3 := by 
    sorry
  sorry

end milk_needed_to_bake_8_cookies_l372_372897


namespace tan_beta_eq_l372_372854

theorem tan_beta_eq
  (a b : ℝ)
  (α β γ : ℝ)
  (h1 : (a + b) / (a - b) = (Real.tan ((α + β) / 2)) / (Real.tan ((α - β) / 2))) 
  (h2 : (α + β) / 2 = 90 - γ / 2) 
  (h3 : (α - β) / 2 = 90 - (β + γ / 2)) 
  : Real.tan β = (2 * b * Real.tan (γ / 2)) / ((a + b) * (Real.tan (γ / 2))^2 + (a - b)) :=
by
  sorry

end tan_beta_eq_l372_372854


namespace matrix_det_example_l372_372617

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l372_372617


namespace solve_for_a_l372_372005

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372005


namespace largest_arithmetic_mean_two_digit_pairs_l372_372285

noncomputable def largest_arithmetic_mean : ℕ := 75

theorem largest_arithmetic_mean_two_digit_pairs :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a > b ∧
  (a + b) / 2 = (25 * (Int.natAbs ((Real.sqrt (a * b)).floor) + 1)) / 24 ∧
  ∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x > y ∧
                  (x + y) / 2 = (25 * (Int.natAbs ((Real.sqrt (x * y)).floor) + 1)) / 24 →
                  (x + y) / 2 ≤ largest_arithmetic_mean :=
begin
  sorry
end

end largest_arithmetic_mean_two_digit_pairs_l372_372285


namespace bake_sale_money_raised_correct_l372_372973

def bake_sale_money_raised : Prop :=
  let chocolate_chip_cookies_baked := 4 * 12
  let oatmeal_raisin_cookies_baked := 6 * 12
  let regular_brownies_baked := 2 * 12
  let sugar_cookies_baked := 6 * 12
  let blondies_baked := 3 * 12
  let cream_cheese_swirled_brownies_baked := 5 * 12
  let chocolate_chip_cookies_price := 1.50
  let oatmeal_raisin_cookies_price := 1.00
  let regular_brownies_price := 2.50
  let sugar_cookies_price := 1.25
  let blondies_price := 2.75
  let cream_cheese_swirled_brownies_price := 3.00
  let chocolate_chip_cookies_sold := 0.75 * chocolate_chip_cookies_baked
  let oatmeal_raisin_cookies_sold := 0.85 * oatmeal_raisin_cookies_baked
  let regular_brownies_sold := 0.60 * regular_brownies_baked
  let sugar_cookies_sold := 0.90 * sugar_cookies_baked
  let blondies_sold := 0.80 * blondies_baked
  let cream_cheese_swirled_brownies_sold := 0.50 * cream_cheese_swirled_brownies_baked
  let total_money_raised := 
    chocolate_chip_cookies_sold * chocolate_chip_cookies_price + 
    oatmeal_raisin_cookies_sold * oatmeal_raisin_cookies_price + 
    regular_brownies_sold * regular_brownies_price + 
    sugar_cookies_sold * sugar_cookies_price + 
    blondies_sold * blondies_price + 
    cream_cheese_swirled_brownies_sold * cream_cheese_swirled_brownies_price
  total_money_raised = 397.00

theorem bake_sale_money_raised_correct : bake_sale_money_raised := by
  sorry

end bake_sale_money_raised_correct_l372_372973


namespace maximum_common_roots_l372_372431

noncomputable theory
open Complex Polynomial

def condition (n k : ℕ) (c : Fin n → ℂ) : Prop :=
  n ≥ k^2 - 3 * k + 4 ∧ ∀ i : Fin n, c i * c (Fin.val ⟨n - 1 - i, sorry⟩) = 0

theorem maximum_common_roots (k n : ℕ) (c : Fin n → ℂ) (h : condition n k c) :
∃ f : Polynomial ℂ, f.degree = n - 1 ∧ (∀ i : ℕ, i < n - 1 → coeff f i = c ⟨i, sorry⟩) ∧
(∀ ξ : ℂ, (isRoot f ξ) → ξ^n = 1) → (nat_degree ((f:Polynomial ℂ) * (Polynomial.X^n - 1)) ≤ n - k) :=
sorry

end maximum_common_roots_l372_372431


namespace rectangle_perimeter_l372_372117

-- We first define the side lengths of the squares and their relationships
def b1 : ℕ := 3
def b2 : ℕ := 9
def b3 := b1 + b2
def b4 := 2 * b1 + b2
def b5 := 3 * b1 + 2 * b2
def b6 := 3 * b1 + 3 * b2
def b7 := 4 * b1 + 3 * b2

-- Dimensions of the rectangle
def L := 37
def W := 52

-- Theorem to prove the perimeter of the rectangle
theorem rectangle_perimeter : 2 * L + 2 * W = 178 := by
  -- Proof will be provided here
  sorry

end rectangle_perimeter_l372_372117


namespace polynomial_degree_n_l372_372640

noncomputable
def exists_polynomial_degree (n : ℕ) : Prop :=
  ∃ (f : (ℝ → ℝ)) [IsPolynomial f], degree f = n ∧ ∀ x, f(x^2 + 1) = f(x)^2 + 1

theorem polynomial_degree_n (n : ℕ) :
  exists_polynomial_degree n ↔ ∃ (k : ℕ), n = 2^k :=
sorry

end polynomial_degree_n_l372_372640


namespace birds_on_fence_l372_372863

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l372_372863


namespace isosceles_triangle_l372_372775

def shape_of_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : Prop :=
  A = B

theorem isosceles_triangle {A B C : Real} (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  shape_of_triangle A B C h := 
  sorry

end isosceles_triangle_l372_372775


namespace binomial_sum_real_part_l372_372982

theorem binomial_sum_real_part :
  (1 / 2 ^ 1988) * (∑ n in Finset.range 995, (-3) ^ n * Nat.choose 1988 (2 * n)) = -1 / 2 :=
by
  sorry

end binomial_sum_real_part_l372_372982


namespace find_x_l372_372738

theorem find_x (x y z : ℝ) (h1 : x ≠ 0) 
  (h2 : x / 3 = z + 2 * y ^ 2) 
  (h3 : x / 6 = 3 * z - y) : 
  x = 168 :=
by
  sorry

end find_x_l372_372738


namespace complex_magnitude_example_l372_372742

theorem complex_magnitude_example (z : ℂ) (h : z = 1 + Complex.i) : Complex.abs (z^2 - 2*z) = 2 := 
by 
  rw [h]
  sorry

end complex_magnitude_example_l372_372742


namespace chocolates_difference_l372_372089

theorem chocolates_difference :
  ∀ (Robert Nickel : ℕ), Robert = 7 → Nickel = 5 → Robert - Nickel = 2 := 
by
  intros Robert Nickel hRobert hNickel
  rw [hRobert, hNickel]
  rfl

end chocolates_difference_l372_372089


namespace problem_solution_l372_372336

-- Definition of i as the imaginary unit
def i : ℂ := complex.I

-- Given definitions
def z (a : ℝ) : ℂ := (1 + a * i) * (1 - i)
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Statement to prove
theorem problem_solution :
  ∃ a : ℝ, purely_imaginary (z a) ∧ a = -1 ∧ complex.abs (z a + i) = 1 :=
by {
  sorry
}

end problem_solution_l372_372336


namespace regression_line_is_y_eq_x_plus_1_l372_372786

def Point : Type := ℝ × ℝ

def A : Point := (1, 2)
def B : Point := (2, 3)
def C : Point := (3, 4)
def D : Point := (4, 5)

def points : List Point := [A, B, C, D]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.foldr (fun x acc => x + acc) 0) / lst.length

noncomputable def regression_line (pts : List Point) : ℝ → ℝ :=
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  fun x : ℝ => x + 1

theorem regression_line_is_y_eq_x_plus_1 :
  regression_line points = fun x => x + 1 := sorry

end regression_line_is_y_eq_x_plus_1_l372_372786


namespace dot_product_example_l372_372306

def vec (k : ℤ) : (ℝ × ℝ) := (Real.cos (k * Real.pi / 6), Real.sin (k * Real.pi / 6) + Real.cos (k * Real.pi / 6))

theorem dot_product_example :
  let a2015 := vec 2015
  let a2016 := vec 2016
  a2015.1 * a2016.1 + a2015.2 * a2016.2 = ℝ.sqrt 3 - 1 / 2 := by
  sorry

end dot_product_example_l372_372306


namespace sum_of_a_and_b_l372_372325

theorem sum_of_a_and_b (a b : ℝ) (h : 2^b = 2^(6 - a)) : a + b = 6 := by
  sorry

end sum_of_a_and_b_l372_372325


namespace find_a_l372_372061

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372061


namespace triangle_square_area_ratio_l372_372787

theorem triangle_square_area_ratio (x : ℝ)
  (A B C D M N : ℝ → ℝ)
  (h_square : quadrilateral A B C D)
  (h_AB : dist A B = x) (h_AD : dist A D = x)
  (h_A_midpoint : is_midpoint M A B) 
  (h_C_midpoint : is_midpoint N C D) :
  (area (triangle A M N)) / (area (square A B C D)) = 1 / 4 :=
sorry

end triangle_square_area_ratio_l372_372787


namespace solve_for_a_l372_372004

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372004


namespace vertex_edges_labels_l372_372406

-- Definitions reflecting the problem conditions.
-- Assuming we have vertices labeled from 1 to 8.
def vertices : Type := Fin 8

-- Beautiful face definition.
def is_beautiful (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

-- Cube representation with numbered vertices and edges.
structure cube :=
  (faces : Fin 6 → (vertices × vertices × vertices × vertices))
  (vertex_label : vertices → ℕ)
  (beautiful_faces_count : ∀ i : Fin 6, is_beautiful (vertex_label i.1.1) (vertex_label i.1.2) (vertex_label i.2.1) (vertex_label i.2.2))

-- Given a vertex labeled 6.
axiom vertex_with_6 : {v // cube.vertex_label v = 6}

-- Proof statement.
theorem vertex_edges_labels (v : {v // cube.vertex_label v = 6}) : 
  ∃ a b c : vertices, (v.val, a) ∈ cube.faces ∧ (v.val, b) ∈ cube.faces ∧ (v.val, c) ∈ cube.faces ∧ 
  cube.vertex_label a = 2 ∧ cube.vertex_label b = 3 ∧ cube.vertex_label c = 5 :=
by
  admit -- Proof goes here.

end vertex_edges_labels_l372_372406


namespace smallest_number_of_eggs_l372_372178

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372178


namespace ratio_of_triangle_TPQ_to_trapezoid_PQRS_l372_372793

-- Definitions of given lengths
def PQ : ℝ := 5
def RS : ℝ := 20

-- Define the function to calculate the ratio of areas
def ratio_of_areas (area_TPQ area_PQRS : ℝ) : ℝ := area_TPQ / area_PQRS

-- Theorem stating the ratio of the areas given the conditions
theorem ratio_of_triangle_TPQ_to_trapezoid_PQRS (area_TPQ area_PQRS : ℝ) 
  (hPQ : PQ = 5) (hRS : RS = 20) 
  (H : (RS / PQ) ^ 2 = 16) 
  (H2 : area_PQRS + area_TPQ = 16 * area_TPQ) : 
  ratio_of_areas area_TPQ area_PQRS = 1 / 15 := by 
  sorry

end ratio_of_triangle_TPQ_to_trapezoid_PQRS_l372_372793


namespace major_axis_length_l372_372218

/-- Given:
1. A right circular cylinder with radius 2.
2. The resulting ellipse has a major axis that is 40% longer than its minor axis.
Prove: The length of the major axis is 5.6. -/
theorem major_axis_length (r : ℝ) (H_radius : r = 2)
  (H_major_minor_ratio : ∀ (minor major : ℝ), major = 1.4 * minor) :
  ∃ (major : ℝ), major = 5.6 :=
by
  have minor := 2 * r
  have H_minor : minor = 4 := by rw [H_radius] simp
  have major := 1.4 * minor
  use major
  have H_major := H_major_minor_ratio minor major
  dsimp at H_major
  rw [H_minor] at H_major
  exact H_major
  sorry

end major_axis_length_l372_372218


namespace sum_faces_edges_vertices_square_pyramid_l372_372542

theorem sum_faces_edges_vertices_square_pyramid : 
  let F := 5 in
  let E := 8 in
  let V := 5 in
  F + E + V = 18 := by
  sorry

end sum_faces_edges_vertices_square_pyramid_l372_372542


namespace sum_of_angles_is_right_angle_l372_372955

-- Define the mast height and shadows
def mast_height : ℝ := 1
def shadow_length1 : ℝ := 1
def shadow_length2 : ℝ := 2
def shadow_length3 : ℝ := 3

-- Define the angles of incidence
def angle_ACD : ℝ := real.arctan (mast_height / shadow_length1)
def angle_AEB : ℝ := real.arctan (mast_height / shadow_length2)
def angle_ADB : ℝ := real.arctan (mast_height / shadow_length3)

-- The theorem to be proved
theorem sum_of_angles_is_right_angle :
  angle_ACD + angle_AEB + angle_ADB = real.pi / 2 := sorry

end sum_of_angles_is_right_angle_l372_372955


namespace acute_angle_range_l372_372730

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (m-2, m+3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (2m+1, m-2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem acute_angle_range (m : ℝ) :
  (2 < m) ∨ 
  (m < ((-11 - 5 * Real.sqrt 5) / 2)) ∨ 
  (((-11 + 5 * Real.sqrt 5) / 2) < m ∧ m < (-4/3)) ↔
  (dot_product (vector_a m) (vector_b m) > 0) ∧ 
  ¬(collinear (vector_a m) (vector_b m)) := sorry

end acute_angle_range_l372_372730


namespace sum_first_five_terms_l372_372319

theorem sum_first_five_terms (a1 a2 a3 : ℝ) (S5 : ℝ) 
  (h1 : a1 * a3 = 8 * a2)
  (h2 : (a1 + a2) = 24) :
  S5 = 31 :=
sorry

end sum_first_five_terms_l372_372319


namespace arithmetic_mean_of_fractions_l372_372355

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 5
  let b := (5 : ℚ) / 7
  (a + b) / 2 = (23 : ℚ) / 35 := 
by 
  sorry 

end arithmetic_mean_of_fractions_l372_372355


namespace partition_exists_l372_372813

variables {V : Type*} [fintype V] {ε : ℝ} {n k : ℕ} 
          (C : fin k.succ → finset V) (P P' : finset (finset V))
          (c : ℕ)

-- Definitions as per conditions
def partition_of_V (P : finset (finset V)) (V : finset V) : Prop := P.sup id = V
def exceptional_set (C0 : finset V) (ε : ℝ) (n : ℕ) : Prop := C0.card ≤ (ε * n).to_nat
def equal_sized_non_exc_sets (C : fin k.succ → finset V) (c : ℕ) : Prop := 
  ∀ i, 1 ≤ i → i < k.succ → (C i).card = c
def epsilon_regular (P : finset (finset V)) (ε : ℝ) : Prop := sorry  -- Placeholder definition for ε-regularity
def q_function (P : finset (finset V)) : ℝ := sorry  -- Placeholder definition for q-function

-- Conditions
variables (C₀ : finset V) (hε : 0 < ε ∧ ε ≤ 1/4)
          (P_h : partition_of_V P)
          (P_not_ε_reg : ¬ epsilon_regular P ε)
          (hC₀_card : exceptional_set C₀ ε n)
          (hequal : equal_sized_non_exc_sets C c)

-- To prove:
theorem partition_exists (P : finset (finset V)) :
  ∃ (P' : finset (finset V)), 
    partition_of_V P' ∧ exceptional_set (P'.inf id) ε n ∧ 
    ∃ (l : ℕ), k ≤ l ∧ l ≤ k * 4^k ∧ 
    equal_sized_non_exc_sets (λ i, P'.inf' id) c ∧ 
    q_function P' ≥ q_function P + ε^5 / 2 :=
sorry

end partition_exists_l372_372813


namespace three_digit_max_product_l372_372154

def max_product_3_digit (a b c d e : ℕ) : Prop :=
  (perm [a, b, c, d, e] [3, 5, 7, 8, 9]) →
  ∃ x y : ℕ, 
  100 * a + 10 * b + c = x ∧ 
  10 * d + e = y ∧ 
  (x * y <= 953 * 87)

theorem three_digit_max_product : 
  (∃ (a b c d e : ℕ), max_product_3_digit a b c d e) :=
sorry

end three_digit_max_product_l372_372154


namespace difference_of_squares_multiple_of_20_l372_372195

theorem difference_of_squares_multiple_of_20 (a b : ℕ) (h1 : a > b) (h2 : a + b = 10) (hb : b = 10 - a) : 
  ∃ k : ℕ, (9 * a + 10)^2 - (100 - 9 * a)^2 = 20 * k :=
by
  sorry

end difference_of_squares_multiple_of_20_l372_372195


namespace zeros_sum_lt_2_l372_372818

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem zeros_sum_lt_2 (a b c : ℝ) (f : ℝ → ℝ) (x1 x2 x3 : ℝ) :
  a ≠ 0 →
  6 * a + b = 0 →
  f(x1) - x1 * exp (-x1) = 0 →
  f(x2) - x2 * exp (-x2) = 0 →
  f(x3) - x3 * exp (-x3) = 0 →
  0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 3 →
  (∀ x, f x = a * x^3 + b * x^2 + c * x) →
  f 1 = 4 * a →
  x1 + x2 + x3 < 2 :=
begin
  -- Skipping the proof
  sorry
end

end zeros_sum_lt_2_l372_372818


namespace chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l372_372239

variables (Bodyguards : Type)
variables (U S T : Bodyguards → Prop)

-- Conditions
axiom cond1 : ∀ x, (T x → (S x → U x))
axiom cond2 : ∀ x, (S x → (U x ∨ T x))
axiom cond3 : ∀ x, (¬ U x ∧ T x → S x)

-- To prove
theorem chess_players_swim : ∀ x, (S x → U x) := by
  sorry

theorem not_every_swimmer_plays_tennis : ¬ ∀ x, (U x → T x) := by
  sorry

theorem tennis_players_play_chess : ∀ x, (T x → S x) := by
  sorry

end chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l372_372239


namespace range_of_2_pow_x_add_1_l372_372883

theorem range_of_2_pow_x_add_1 :
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → 1 ≤ 2^(x + 1) ∧ 2^(x + 1) ≤ 4 :=
by
  intros x hx
  sorry

end range_of_2_pow_x_add_1_l372_372883


namespace solution_set_of_inequality_l372_372885

theorem solution_set_of_inequality :
  {x : ℝ | x^2 * (x - 4) ≥ 0} = {x : ℝ | x = 0 ∨ x ≥ 4} :=
by
  sorry

end solution_set_of_inequality_l372_372885


namespace ellipse_standard_equation_l372_372693

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : a > b > 0)
  (h2 : c / a = (Real.sqrt 3) / 3)
  (h3 : 4 * a = 4 * Real.sqrt 3)
  (h4 : a ^ 2 = b ^ 2 + c ^ 2) :
  (∃ x y : ℝ, (x^2 / 3) + (y^2 / 2) = 1) :=
by
  sorry

end ellipse_standard_equation_l372_372693


namespace regular_price_of_tire_l372_372962

theorem regular_price_of_tire (p : ℝ) (h : 2 * p + p / 2 = 270) : p = 108 :=
sorry

end regular_price_of_tire_l372_372962


namespace no_m_such_that_S_m_plus_1_eq_4_S_m_l372_372678

def least_common_multiple_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).tail.lcm

theorem no_m_such_that_S_m_plus_1_eq_4_S_m :
  ¬ ∃ m : ℕ, least_common_multiple_up_to (m + 1) = 4 * least_common_multiple_up_to m :=
by
  sorry

end no_m_such_that_S_m_plus_1_eq_4_S_m_l372_372678


namespace relationship_of_exponents_l372_372307

theorem relationship_of_exponents (m p r s : ℝ) (u v w t : ℝ) (h1 : m^u = r) (h2 : p^v = r) (h3 : p^w = s) (h4 : m^t = s) : u * v = w * t :=
by
  sorry

end relationship_of_exponents_l372_372307


namespace bus_ticket_problem_l372_372203

variables (x y : ℕ)

theorem bus_ticket_problem (h1 : x + y = 99) (h2 : 2 * x + 3 * y = 280) : x = 17 ∧ y = 82 :=
by
  sorry

end bus_ticket_problem_l372_372203


namespace smallest_number_of_eggs_l372_372174

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372174


namespace population_growth_l372_372114

theorem population_growth :
  ∃ (pop : ℕ), 
    pop = 319686 ∧
    initial_population = 240000 ∧ 
    growth_rate_first_five_years = 0.025 ∧ 
    growth_rate_next_four_years = 0.03 ∧ 
    growth_rate_last_three_years = 0.015 ∧ 
    population_after_twelve_years initial_population growth_rate_first_five_years growth_rate_next_four_years growth_rate_last_three_years = pop := 
sorry

end population_growth_l372_372114


namespace find_sixth_term_l372_372504

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l372_372504


namespace calc_expression_l372_372244

theorem calc_expression : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end calc_expression_l372_372244


namespace student_count_incorrect_l372_372588

theorem student_count_incorrect :
  ∀ k : ℕ, 2012 ≠ 18 + 17 * k :=
by
  intro k
  sorry

end student_count_incorrect_l372_372588


namespace milkman_A_rent_share_l372_372566

theorem milkman_A_rent_share : 
  let A_cows := 24
  let A_months := 3
  let B_cows := 10
  let B_months := 5
  let C_cows := 35
  let C_months := 4
  let D_cows := 21
  let D_months := 3
  let total_rent := 3250
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let fraction_A := A_cow_months / total_cow_months
  let A_rent_share := total_rent * fraction_A
  A_rent_share = 720 := 
by
  sorry

end milkman_A_rent_share_l372_372566


namespace collinear_MNO_l372_372149

variable (O₁ O₂ P Q O A B C D M N : Point)
variable (circle₁ circle₂ : Circle O₁ O₂)
variable [NonEqualRadiuses] : EqualRadii(circle₁, circle₂)
variable [Intersects] : Intersects(circle₁, circle₂)
variable [NotInsideCommonArea] : NotInsideCommonArea(O₁, O₂)
variable [SecantA_B] : SecantThrough(circle₁, circle₂, P, A, B)
variable [SecantC_D] : SecantThrough(circle₁, circle₂, P, C, D)
variable [MidpointP_Q] : Midpoint(P, Q, O)
variable [MidpointA_D] : Midpoint(A, D, M)
variable [MidpointB_C] : Midpoint(B, C, N)
variable [NotCoincide] : ¬Coincide(O, M) ∧ ¬Coincide(O, N)

theorem collinear_MNO :
  Collinear(M, N, O) :=
sorry

end collinear_MNO_l372_372149


namespace abs_z2_minus_2z_eq_2_l372_372751

theorem abs_z2_minus_2z_eq_2 (z : ℂ) (h : z = 1 + complex.I) : abs (z^2 - 2*z) = 2 := by
  sorry

end abs_z2_minus_2z_eq_2_l372_372751


namespace minimum_possible_area_l372_372970

theorem minimum_possible_area (l w l_min w_min : ℝ) (hl : l = 5) (hw : w = 7) 
  (hl_min : l_min = l - 0.5) (hw_min : w_min = w - 0.5) : 
  l_min * w_min = 29.25 :=
by
  sorry

end minimum_possible_area_l372_372970


namespace rational_zero_quadratic_roots_l372_372193

-- Part 1
theorem rational_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 :=
sorry

-- Part 2
theorem quadratic_roots (k : ℝ) (h : k ≠ 0) (x1 x2 : ℝ)
  (h1 : 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2 : 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h3 : x1 ≠ x2) 
  (h4 : x1^2 + x2^2 - 2 * x1 * x2 = 0.5) : k = -2 :=
sorry

end rational_zero_quadratic_roots_l372_372193


namespace construct_triangle_with_conditions_l372_372628

-- Define the given points P, Q, R
variables (P Q R : Point)

-- Define the problem: Construct a triangle satisfying the conditions
theorem construct_triangle_with_conditions
    (hPQR : ∃ (C : Point) (circABC : Circle) (alt_C : Line) (bis_C : Line) (med_C : Line),
        (P ∈ circABC) ∧ (Q ∈ circABC) ∧ (R ∈ circABC) ∧
        (alt_C ∩ circABC = {P}) ∧
        (bis_C ∩ circABC = {Q}) ∧
        (med_C ∩ circABC = {R}) ∧
        (is_altitude C alt_C) ∧ 
        (is_angle_bisector C bis_C) ∧ 
        (is_median C med_C) ∧ 
        (circABC = circumcircle (triangle ABC))): ∃ (A B C : Point), is_triangle A B C :=
by
  sorry

end construct_triangle_with_conditions_l372_372628


namespace cos_60_eq_half_l372_372133

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l372_372133


namespace John_can_finish_work_alone_in_48_days_l372_372420

noncomputable def John_and_Roger_can_finish_together_in_24_days (J R: ℝ) : Prop :=
  1 / J + 1 / R = 1 / 24

noncomputable def John_finished_remaining_work (J: ℝ) : Prop :=
  (1 / 3) / (16 / J) = 1

theorem John_can_finish_work_alone_in_48_days (J R: ℝ) 
  (h1 : John_and_Roger_can_finish_together_in_24_days J R) 
  (h2 : John_finished_remaining_work J):
  J = 48 := 
sorry

end John_can_finish_work_alone_in_48_days_l372_372420


namespace ellipse_eccentricity_l372_372261

theorem ellipse_eccentricity (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2) : (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end ellipse_eccentricity_l372_372261


namespace expected_pairs_in_same_row_or_column_l372_372835

noncomputable def grid := Finset (Fin 49)

-- Define a random arrangement of numbers in a 7x7 grid
def random_arrangement (g: grid) : Prop :=
  g.card = 49 -- All numbers from 1 to 49 are present

-- Define the probability that a given pair of numbers is in the same row or column
def same_row_or_column_probability := 1 / 4

-- Define the expected number of pairs
theorem expected_pairs_in_same_row_or_column :
  let total_pairs := (49.choose 2) in
  let probability_same_in_both := same_row_or_column_probability ^ 2 in
  let expected_value := total_pairs * probability_same_in_both in
  expected_value = 73.5 := by
begin
  let total_pairs := 49.choose 2,
  let probability_same_in_both := (1 / 4) ^ 2,
  let expected_value := total_pairs * probability_same_in_both,
  have total_pairs_eq : total_pairs = 1176 := by norm_num,
  have probability_eq : probability_same_in_both = 1 / 16 := by norm_num,
  have expected_value_eq : expected_value = 73.5 := by norm_num,
  exact expected_value_eq,
end

end expected_pairs_in_same_row_or_column_l372_372835


namespace inequality_proof_l372_372826

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ real.sqrt (a * b) + real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l372_372826


namespace area_of_equilateral_triangle_l372_372379

noncomputable def lines_forming_equilateral_triangle (t : ℝ) : Prop :=
  let l1 := λ (x y: ℝ), x * Real.cos t + (y + 1) * Real.sin t = 2
  let l2 := λ (x y: ℝ), x * Real.cos (t + 2 * Real.pi / 3) + (y + 1) * Real.sin (t + 2 * Real.pi / 3) = 2
  let l3 := λ (x y: ℝ), x * Real.cos (t + 4 * Real.pi / 3) + (y + 1) * Real.sin (t + 4 * Real.pi / 3) = 2
  ∃ A B C : ℝ × ℝ, l1 A.1 A.2 ∧ l2 B.1 B.2 ∧ l3 C.1 C.2 ∧
    eq_sub_eq_add_sub_eq A B C 4

theorem area_of_equilateral_triangle (t : ℝ) (h : lines_forming_equilateral_triangle t) : ∃ A : ℝ, A = 12 * Real.sqrt 3 :=
sorry

def eq_sub_eq_add_sub_eq (A B C : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

end area_of_equilateral_triangle_l372_372379


namespace product_of_solutions_of_t_squared_eq_49_l372_372666

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372666


namespace unique_alpha_l372_372811

theorem unique_alpha (α : ℝ) (r : ℕ) (hα : α ≥ 1) (hR : 0 ≤ r < 2021) :
  (∀ n : ℕ, n ∈ (λ k : ℕ, floor (k * α)) '' {k : ℕ | 0 < k} ↔ ∀ m : ℕ, m ≡ r [MOD 2021] → m ∉ (λ k : ℕ, floor (k * α)) '' {k : ℕ | 0 < k}) ↔ α = 2021 / 2020 := 
sorry

end unique_alpha_l372_372811


namespace equal_acquaintances_l372_372888

theorem equal_acquaintances (n : ℕ)
  (H1 : ∀ (x y : Fin n), ¬(x = y) → ∃! (z : Fin n), (¬ (x = z) ∧ ¬ (y = z) ∧ (¬ ∃ (a : Fin n), (a ≠ x ∧ a ≠ y ∧ z = a)) ∧ ∃ (a b : Fin n), (¬ (x = a) ∧ ¬ (y = a) ∧ ¬ (a = b) ∧ ¬ (x = b) ∧ ¬ (y = b) ∧ z = a ∧ z = b ∧ z ≠ x ∧ z ≠ y))) 
  (H2 : ∀ (x y z : Fin n), (¬ (x = y) ∧ ¬ (y = z) ∧ ¬ (x = z)) → (¬ ∃ (a : Fin n), (a ≠ x ∧ a ≠ y ∧ a ≠ z ∧ (a = y ∨ a = z)))) :
  ∀(x y : Fin n), ∃ (m : ℕ), (∀ z : Fin n, m = (card {w : Fin n | w ≠ z ∧ z ≠ w})) :=
by
  sorry

end equal_acquaintances_l372_372888


namespace relationship_between_coefficients_l372_372162

theorem relationship_between_coefficients
  (b c : ℝ)
  (h_discriminant : b^2 - 4 * c ≥ 0)
  (h_root_condition : ∃ x1 x2 : ℝ, x1^2 = -x2 ∧ x1 + x2 = -b ∧ x1 * x2 = c):
  b^3 - 3 * b * c - c^2 - c = 0 :=
by
  sorry

end relationship_between_coefficients_l372_372162


namespace find_angle_A_condition1_find_angle_A_condition2_find_area_range_l372_372410

variables {a b c : ℝ} (A B C : ℝ)
variables (triangle : Prop) (acute : Prop)

noncomputable def angle_A_condition1 (h1 : (b + c) * (Real.sin B + Real.sin C) = a * Real.sin A + 3 * b * Real.sin C) : Prop :=
  A = Real.pi / 3

noncomputable def angle_A_condition2 (h2 : Real.cos (Real.pi / 2 + A) ^ 2 + Real.cos A = 5 / 4) : Prop :=
  A = Real.pi / 3

theorem find_angle_A_condition1 (h1 : (b + c) * (Real.sin B + Real.sin C) = a * Real.sin A + 3 * b * Real.sin C) :
  angle_A_condition1 A h1 := 
sorry

theorem find_angle_A_condition2 (h2 : Real.cos (Real.pi / 2 + A) ^ 2 + Real.cos A = 5 / 4) :
  angle_A_condition2 A h2 :=
sorry

noncomputable def area_range (c_eq_1 : c = 1) (acute_triangle : acute ∧ triangle) : Prop :=
  ∃ b : ℝ, b ∈ Set.Ioo (1/2) 2 ∧ (√3/8 < (√3/4) * b ∧ (√3/4) * b < √3/2)

theorem find_area_range (h3 : c = 1) (h4 : acute ∧ triangle) :
  area_range A h3 h4 :=
sorry

end find_angle_A_condition1_find_angle_A_condition2_find_area_range_l372_372410


namespace graph_intersects_x_axis_at_three_points_l372_372708

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x + 1 else real.log x / real.log 2

theorem graph_intersects_x_axis_at_three_points :
  {x : ℝ | f (f x) - 1 = 0}.finite.to_finset.card = 3 :=
by
  sorry

end graph_intersects_x_axis_at_three_points_l372_372708


namespace expected_value_and_variance_of_Y_l372_372729
open Probability

variables {Ω : Type*} {X Y : Ω → ℝ}

-- Conditions
variables (h1 : ∀ ω, X ω + 2 * Y ω = 4)
          (h2 : ∀ s, MeasureTheory.MeasureSpace.has_distribution (λ ω, X ω) (MeasureTheory.Measure.norm N 1 (2 ^ 2)))

-- Expected value and variance of Y
theorem expected_value_and_variance_of_Y :
  E[Y] = 3 / 2 ∧ var Y = 1 :=
sorry

end expected_value_and_variance_of_Y_l372_372729


namespace ellipse_eccentricity_l372_372036

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l372_372036


namespace find_z_l372_372136

variable {x y z : ℝ}

def condition1 : Prop := x = (1/3) * y
def condition2 : Prop := y = (1/4) * z
def condition3 : Prop := x + y = 16

theorem find_z (h1 : condition1) (h2 : condition2) (h3 : condition3) : z = 48 := by
  sorry

end find_z_l372_372136


namespace coincide_foci_l372_372337

variable (m n : ℝ)

def ellipse (m : ℝ) := ∀ x y : ℝ, x^2 / m^2 + y^2 = 1
def hyperbola (n : ℝ) := ∀ x y : ℝ, x^2 / n^2 - y^2 = 1

theorem coincide_foci {m n : ℝ} (h₁ : m > 1) (h₂ : n > 0)
  (condition : m^2 - 1 = n^2 + 1) : m > n ∧ (sqrt (m^2 - 1) / m) * (sqrt (m^2 - 1) / n) > 1 :=
by
  sorry

end coincide_foci_l372_372337


namespace find_a_l372_372069

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l372_372069


namespace number_of_such_integers_is_5_l372_372819

def f (x : ℤ) : ℤ :=
  x^2 + 5 * x + 5

def S : Set ℤ :=
  {n | 0 ≤ n ∧ n ≤ 20}

def divisible_by_5 (n : ℤ) :=
  n % 5 = 0

theorem number_of_such_integers_is_5 :
  {n ∈ S | divisible_by_5 (f n)}.card = 5 :=
  sorry

end number_of_such_integers_is_5_l372_372819


namespace solve_for_a_l372_372002

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372002


namespace sum_first_n_terms_arithmetic_sequence_eq_l372_372817

open Nat

noncomputable def sum_arithmetic_sequence (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if h: n = 0 then 0 else n * a₁ + (n * (n - 1) * d) / 2

theorem sum_first_n_terms_arithmetic_sequence_eq 
  (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) 
  (h₀ : d ≠ 0)
  (h₁ : a₁ = 4)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₆ = a₁ + 5 * d)
  (h₄ : a₃^2 = a₁ * a₆) :
  sum_arithmetic_sequence a₁ a₃ a₆ d n = (n^2 + 7 * n) / 2 := 
by
  sorry

end sum_first_n_terms_arithmetic_sequence_eq_l372_372817


namespace hexagon_interior_angles_sum_l372_372517

theorem hexagon_interior_angles_sum (n : ℕ) (hn : n = 6) : (n - 2) * 180 = 720 :=
by
  rw [hn]
  norm_num

end hexagon_interior_angles_sum_l372_372517


namespace find_num_20_paise_coins_l372_372189

def num_20_paise_coins (x y : ℕ) : Prop :=
  x + y = 334 ∧ 20 * x + 25 * y = 7100

theorem find_num_20_paise_coins (x y : ℕ) (h : num_20_paise_coins x y) : x = 250 :=
by
  sorry

end find_num_20_paise_coins_l372_372189


namespace tetrahedron_projection_not_square_with_area_1_tetrahedron_projection_can_be_square_with_area_inv_2019_l372_372509

theorem tetrahedron_projection_not_square_with_area_1 (tetrahedron : Type) 
  (faces : tetrahedron → set (set Point)) 
  (area : set Point → ℝ)
  (h_condition : ∃ (face1 face2 : set Point), faces face1 ∧ faces face2 ∧ area (orth_proj face1) = 1 ∧ is_trapezoid (orth_proj face1)) :
  ∀ facex, faces facex → ¬ (area (orth_proj facex) = 1 ∧ is_square (orth_proj facex)) :=
sorry

theorem tetrahedron_projection_can_be_square_with_area_inv_2019 (tetrahedron : Type) 
  (faces : tetrahedron → set (set Point)) 
  (area : set Point → ℝ)
  (h_condition : ∃ (face1 face2 : set Point), faces face1 ∧ faces face2 ∧ area (orth_proj face1) = 1 ∧ is_trapezoid (orth_proj face1)) :
  ∃ facex, faces facex ∧ area (orth_proj facex) = 1 / 2019 ∧ is_square (orth_proj facex) :=
sorry

end tetrahedron_projection_not_square_with_area_1_tetrahedron_projection_can_be_square_with_area_inv_2019_l372_372509


namespace weight_of_currants_l372_372592

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end weight_of_currants_l372_372592


namespace alternating_series_sum_l372_372257

theorem alternating_series_sum :
  ∑ i in finset.range 100, (if even i then i else -i) = 100 := by 
  sorry

end alternating_series_sum_l372_372257


namespace white_tshirt_cost_l372_372104

-- Define the problem conditions
def total_tshirts : ℕ := 200
def total_minutes : ℕ := 25
def black_tshirt_cost : ℕ := 30
def revenue_per_minute : ℕ := 220

-- Prove the cost of white t-shirts given the conditions
theorem white_tshirt_cost : 
  (total_tshirts / 2) * revenue_per_minute * total_minutes 
  - (total_tshirts / 2) * black_tshirt_cost = 2500
  → 2500 / (total_tshirts / 2) = 25 :=
by
  sorry

end white_tshirt_cost_l372_372104


namespace smallest_even_n_for_reducible_fraction_l372_372602

theorem smallest_even_n_for_reducible_fraction : 
  ∃ (N: ℕ), (N > 2013) ∧ (N % 2 = 0) ∧ (Nat.gcd (15 * N - 7) (22 * N - 5) > 1) ∧ N = 2144 :=
sorry

end smallest_even_n_for_reducible_fraction_l372_372602


namespace decreasing_range_of_a_l372_372875

theorem decreasing_range_of_a :
  ∀ (a : ℝ), (∀ x y, x ∈ Ici (-2) → y ∈ Ioi x → f x ≥ f y) ↔ (2 ≤ a) :=
begin
  sorry
end

end decreasing_range_of_a_l372_372875


namespace absolute_value_z_squared_minus_2z_l372_372755

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem
theorem absolute_value_z_squared_minus_2z : complex.abs (z^2 - 2*z) = 2 := by
  sorry

end absolute_value_z_squared_minus_2z_l372_372755


namespace angle_relationship_l372_372874

variables {VU VW : ℝ} {x y z : ℝ} (h1 : VU = VW) 
          (angle_UXZ : ℝ) (angle_VYZ : ℝ) (angle_VZX : ℝ)
          (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z)

theorem angle_relationship (h1 : VU = VW) (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z) : 
    x = (y - z) / 2 := 
by 
    sorry

end angle_relationship_l372_372874


namespace RS_plus_ST_l372_372489

-- Definitions for the conditions
variables {P Q R S T U V : Type} [HasArea P Q R S T U] [HasArea P Q U V]
variables (PQ QR TU : ℝ)
variables (area_PQRSTU area_PQVU : ℝ)

-- Condition statements
def condition1 : area_PQRSTU = 70 := sorry
def condition2 : PQ = 10 := sorry
def condition3 : QR = 7 := sorry
def condition4 : TU = 6 := sorry
def condition5 : area_PQVU = PQ * QR := sorry 

-- The final proof of the question:
theorem RS_plus_ST :
  PQ = 10 → QR = 7 → TU = 6 → area_PQRSTU = 70 → RS + ST = 6 :=
by
  intros hPQ hQR hTU hArea
  sorry

end RS_plus_ST_l372_372489


namespace average_marks_first_class_l372_372492

theorem average_marks_first_class
  (n1 n2 : ℕ)
  (avg2 : ℝ)
  (combined_avg : ℝ)
  (h_n1 : n1 = 35)
  (h_n2 : n2 = 55)
  (h_avg2 : avg2 = 65)
  (h_combined_avg : combined_avg = 57.22222222222222) :
  (∃ avg1 : ℝ, avg1 = 45) :=
by
  sorry

end average_marks_first_class_l372_372492


namespace common_root_values_l372_372303

def has_common_root (p x : ℝ) : Prop :=
  (x^2 - (p+1)*x + (p+1) = 0) ∧ (2*x^2 + (p-2)*x - p - 7 = 0)

theorem common_root_values :
  (has_common_root 3 2) ∧ (has_common_root (-3/2) (-1)) :=
by {
  sorry
}

end common_root_values_l372_372303


namespace sum_binom_expr_eq_neg_half_l372_372986

def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else (n.factorial) / (k.factorial * (n - k).factorial)

noncomputable def sum_binom_expr : ℝ :=
  (1 / 2 ^ 1988) * ∑ n in Finset.range (995), (-3 : ℝ) ^ n * binom 1988 (2 * n)

theorem sum_binom_expr_eq_neg_half : sum_binom_expr = -1 / 2 := by
  sorry

end sum_binom_expr_eq_neg_half_l372_372986


namespace trailing_zeros_factorial_2010_l372_372167

theorem trailing_zeros_factorial_2010 (n : ℕ) (hn : n = 2010) :
  ∑ k in (range (nat.log 2010 5).succ), n / 5^k = 501 := by
  sorry

end trailing_zeros_factorial_2010_l372_372167


namespace pure_imaginary_iff_real_part_zero_l372_372764

open Complex

theorem pure_imaginary_iff_real_part_zero (a : ℝ) (h_imaginary : (∀ z : ℂ, z = (Complex.mk a 0))
(h_cond: re ((a + Complex.I * 6) / (3 - Complex.I)) = 0) :
  a = 2 := by
  sorry

end pure_imaginary_iff_real_part_zero_l372_372764


namespace triangle_areas_equal_shaded_region_area_l372_372220

-- Define the conditions: ABCD is quadrilateral inscribed in a circle with perpendicular diagonals AC and BD.

-- Problem (a): Prove that triangles AOB and COD have the same area.
theorem triangle_areas_equal (ABCD_inscribed : inscribed_quadrilateral ABCD) 
                             (perpendicular_diagonals : AC ⊥ BD) :
  area (triangle A O B) = area (triangle C O D) :=
sorry

-- Problem (b): Prove that the area of the shaded region is 24 cm^2 given the lengths of the diagonals and the condition of perpendicularity.
theorem shaded_region_area (AC_length : AC = 8) 
                           (BD_length : BD = 6) 
                           (perpendicular_diagonals : AC ⊥ BD) :
  let area_shaded := 1 / 2 * AC * BD
  in area_shaded = 24 :=
sorry

end triangle_areas_equal_shaded_region_area_l372_372220


namespace last_two_digits_of_twentieth_power_l372_372906

theorem last_two_digits_of_twentieth_power (x : ℤ) : 
  let b := x % 10 
  in b ∈ Finset.range 10 →
     (x ^ 20) % 100 = 0 ∨ (x ^ 20) % 100 = 1 ∨ (x ^ 20) % 100 = 25 ∨ (x ^ 20) % 100 = 76 :=
by
  intros b h
  have hb : b = x % 10 := rfl
  sorry

end last_two_digits_of_twentieth_power_l372_372906


namespace tan_of_sin_in_fourth_quadrant_l372_372685

theorem tan_of_sin_in_fourth_quadrant :
  ∀ (α : ℝ), sin α = -sqrt 3 / 2 ∧ (0 < α ∧ α < 2 * π) ∧ (2 * π - α < π / 2) → tan α = -sqrt 3 :=
by
  sorry

end tan_of_sin_in_fourth_quadrant_l372_372685


namespace intersection_A_B_union_A_compB_l372_372322

-- Define the sets A and B
def A : Set ℝ := { x | x^2 + 3 * x - 10 < 0 }
def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the complement of B in the universal set
def comp_B : Set ℝ := { x | ¬ B x }

-- 1. Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_A_B :
  A ∩ B = { x | -5 < x ∧ x ≤ -1 } :=
by 
  sorry

-- 2. Prove that A ∪ (complement of B) = {x | -5 < x ∧ x < 3}
theorem union_A_compB :
  A ∪ comp_B = { x | -5 < x ∧ x < 3 } :=
by 
  sorry

end intersection_A_B_union_A_compB_l372_372322


namespace solve_for_x_l372_372368

theorem solve_for_x :
  ∃ x : ℝ, 5 ^ (Real.logb 5 15) = 7 * x + 2 ∧ x = 13 / 7 :=
by
  sorry

end solve_for_x_l372_372368


namespace local_min_4_l372_372358

def seq (n : ℕ) : ℝ := n^3 - 48 * n + 5

theorem local_min_4 (m : ℕ) (h1 : seq (m-1) > seq m) (h2 : seq (m+1) > seq m) : m = 4 :=
sorry

end local_min_4_l372_372358


namespace earthquake_energy_multiple_l372_372597

theorem earthquake_energy_multiple (E : ℕ → ℝ) (n9 n7 : ℕ)
  (h1 : E n9 = 10 ^ n9) 
  (h2 : E n7 = 10 ^ n7) 
  (hn9 : n9 = 9) 
  (hn7 : n7 = 7) : 
  E n9 / E n7 = 100 := 
by 
  sorry

end earthquake_energy_multiple_l372_372597


namespace prove_KJ_perp_AB_l372_372314

noncomputable def math_problem_statement (a b : ℝ) (A B C D E G H I K J : Point)
  (AB BC AD : Line) : Prop :=
  (let midpoint (P Q : Point) : Point := (P + Q) / 2 in
   let area_triangle (P Q R : Point) : ℝ :=
     0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y)) in
   let parallel (L1 L2 : Line) : Prop := ∃ m n : ℝ, L1.slope = m ∧ L2.slope = n ∧ m = n in
   ∃ A B C D E G H I K J : Point,
   ∃ AB AD BC : Line,
   let midpoint_E := midpoint A D in
   A.x = 0 ∧ A.y = 0 ∧
   B.x = b ∧ B.y = 0 ∧
   C.x = b ∧ C.y = 2 * a ∧
   D.x = 0 ∧ D.y = 2 * a ∧
   E = midpoint_E ∧
   parallel AB (Line.mk E (E + Point.mk b 0)) ∧
   G ∈ (Line.mk E (E + Point.mk b 0)) ∧
   area_triangle G C E = 0.5 * (a^3 / b + a * b) ∧
   H ∈ Line.mk E G ∧
   H.y = 0 ∧ -- Foot of the perpendicular from E to GD
   I ∈ (Line.mk A C) ∧
   (angle (Triangle.mk A C E) = angle (Triangle.mk A E I)) ∧ -- Similar triangles
   K ∈ (Line.mk B H) ∧
   K ∈ (Line.mk I E) ∧
   J ∈ (Line.mk C A) ∧
   J ∈ (Line.mk E H) ∧
   perpendicular (Line.mk K J) (Line.mk A B)) -- Conclusion
\

theorem prove_KJ_perp_AB (a b : ℝ) (A B C D E G H I K J : Point)
  (AB BC AD : Line) : math_problem_statement a b A B C D E G H I K J AB BC AD :=
sorry

end prove_KJ_perp_AB_l372_372314


namespace total_cost_is_correct_l372_372832

namespace MrsBrynleesStore

def original_prices := 
  { shirt := 80,
    jeans := 100,
    leatherJacket := 150,
    dress := 120 }

def quantities := 
  { shirts := 8,
    jeans := 6,
    leatherJackets := 15,
    dresses := 10 }

def price_reduction_rate := 0.25
def additional_discount_threshold := 500
def additional_discount_rate := 0.10
def sales_tax_rate := 0.05

def item_prices_after_reduction : Type :=
  { shirt := original_prices.shirt * (1 - price_reduction_rate),
    jeans := original_prices.jeans * (1 - price_reduction_rate),
    leatherJacket := original_prices.leatherJacket * (1 - price_reduction_rate),
    dress := original_prices.dress * (1 - price_reduction_rate) }

def total_cost_before_discount : ℝ :=
  quantities.shirts * item_prices_after_reduction.shirt +
  quantities.jeans * item_prices_after_reduction.jeans +
  quantities.leatherJackets * item_prices_after_reduction.leatherJacket +
  quantities.dresses * item_prices_after_reduction.dress

def total_cost_after_discount : ℝ :=
  if total_cost_before_discount > additional_discount_threshold then
    total_cost_before_discount * (1 - additional_discount_rate)
  else
    total_cost_before_discount

def total_cost_incl_tax : ℝ := 
  total_cost_after_discount * (1 + sales_tax_rate)

theorem total_cost_is_correct :
  total_cost_incl_tax = 3324.04 := 
sorry

end MrsBrynleesStore

end total_cost_is_correct_l372_372832


namespace find_a_l372_372059

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372059


namespace length_ab_square_l372_372916

theorem length_ab_square (s a : ℝ) (h_square : s = 2 * a) (h_area : 3000 = 1/2 * (s + (s - 2 * a)) * s) : 
  s = 20 * Real.sqrt 15 :=
by
  sorry

end length_ab_square_l372_372916


namespace birds_on_fence_l372_372862

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l372_372862


namespace almond_walnut_ratio_is_5_to_2_l372_372929

-- Definitions based on conditions
variables (A W : ℕ)
def almond_ratio_to_walnut_ratio := A / (2 * W)
def weight_of_almonds := 250
def total_weight := 350
def weight_of_walnuts := total_weight - weight_of_almonds

-- Theorem to prove
theorem almond_walnut_ratio_is_5_to_2
  (h_ratio : almond_ratio_to_walnut_ratio A W = 250 / 100)
  (h_weights : weight_of_walnuts = 100) :
  A = 5 ∧ 2 * W = 2 := by
  sorry

end almond_walnut_ratio_is_5_to_2_l372_372929


namespace total_days_spent_on_island_l372_372804

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l372_372804


namespace monomials_like_terms_l372_372375

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end monomials_like_terms_l372_372375


namespace gcd_of_first_2008_terms_l372_372828

def a (n : ℕ) : ℤ := (2 * n - 1) * (2 * n + 1) * (2 * n + 3)

theorem gcd_of_first_2008_terms :
  (Int.gcd (List.foldr Int.gcd 0 (List.map a (List.range 2008)))) = 3 :=
by
  sorry

end gcd_of_first_2008_terms_l372_372828


namespace final_racers_count_l372_372248

theorem final_racers_count : 
  ∀ (initial_racers eliminated_first third_remaining half_remaining : ℕ),
    initial_racers = 100 →
    eliminated_first = 10 →
    third_remaining = (initial_racers - eliminated_first) / 3 →
    half_remaining = (initial_racers - eliminated_first - third_remaining) / 2 →
  initial_racers - eliminated_first - third_remaining - half_remaining = 30 :=
begin
  intros initial_racers eliminated_first third_remaining half_remaining,
  assume h1 : initial_racers = 100,
  assume h2 : eliminated_first = 10,
  assume h3 : third_remaining = (initial_racers - eliminated_first) / 3,
  assume h4 : half_remaining = (initial_racers - eliminated_first - third_remaining) / 2,
  sorry
end

end final_racers_count_l372_372248


namespace find_AB_l372_372865

theorem find_AB (S α γ : ℝ) 
  (h1 : S > 0) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : 0 < γ ∧ γ < π - α) 
  : ∃ AB, AB = sqrt (2 * S * sin γ / (sin (α + γ) * sin α)) :=
sorry

end find_AB_l372_372865


namespace find_sixth_term_l372_372505

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l372_372505


namespace relationship_M_N_K_l372_372700

-- Definitions based on conditions
def f (x : ℝ) : ℝ := if x > 0 then log x else log (-x)

def M : ℝ := f (-real.pi)
def N : ℝ := f real.exp 1
def K : ℝ := f 2

-- Lean statement to prove the relationship
theorem relationship_M_N_K : M > N ∧ N > K :=
by
  sorry

end relationship_M_N_K_l372_372700


namespace min_sum_real_possible_sums_int_l372_372310

-- Lean 4 statement for the real numbers case
theorem min_sum_real (x y : ℝ) (hx : x + y + 2 * x * y = 5) (hx_pos : x > 0) (hy_pos : y > 0) :
  x + y ≥ Real.sqrt 11 - 1 := 
sorry

-- Lean 4 statement for the integers case
theorem possible_sums_int (x y : ℤ) (hx : x + y + 2 * x * y = 5) :
  x + y = 5 ∨ x + y = -7 :=
sorry

end min_sum_real_possible_sums_int_l372_372310


namespace scientists_arrival_probability_l372_372901

open Real

theorem scientists_arrival_probability (x y z : ℕ) (n : ℝ) (h : z ≠ 0)
  (hz : ¬ ∃ p : ℕ, Nat.Prime p ∧ p ^ 2 ∣ z)
  (h1 : n = x - y * sqrt z)
  (h2 : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 120 ∧ 0 ≤ b ∧ b ≤ 120 ∧
    |a - b| ≤ n)
  (h3 : (120 - n)^2 / (120 ^ 2) = 0.7) :
  x + y + z = 202 := sorry

end scientists_arrival_probability_l372_372901


namespace exists_triangle_with_area_at_least_one_l372_372812

open MeasureTheory

-- Assume a definition of rectangular surface in the ambient space
structure RectangularSurface where
  area : ℝ
  projection : Set ℝ

-- Define the hypothesis that there is a family of such surfaces
def disjoint_family_of_rectangular_surfaces (R : ℕ → RectangularSurface) (n : ℕ) : Prop :=
  (∀ i j, i ≠ j → Set.disjoint (R i).projection (R j).projection) ∧ 
  (∑ i in Finset.range n, (R i).area = 4) ∧ 
  (R 1).projection = Interval a b -- Assuming R_1 has the same projection as ∪_i R_i for simplicity

theorem exists_triangle_with_area_at_least_one 
  {n : ℕ} (R : ℕ → RectangularSurface) 
  (a b : ℝ) (-- conditions)
  (h_family : disjoint_family_of_rectangular_surfaces R n) : 
  ∃ (A B C : ℝ × ℝ), A ∈ (⋃ i, (R i).projection) ∧ B ∈ (⋃ i, (R i).projection) ∧ C ∈ (⋃ i, (R i).projection) ∧ 
  (let area := (1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|) 
  in area ≥ 1) :=
by
  sorry


end exists_triangle_with_area_at_least_one_l372_372812


namespace even_natural_number_factors_count_l372_372645

def is_valid_factor (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 3 ∧ 
  0 ≤ b ∧ b ≤ 2 ∧ 
  0 ≤ c ∧ c ≤ 2 ∧ 
  a + b + c ≤ 4

noncomputable def count_valid_factors : ℕ :=
  Nat.card { x : ℕ × ℕ × ℕ // is_valid_factor x.1 x.2.1 x.2.2 }

theorem even_natural_number_factors_count : count_valid_factors = 15 := 
  sorry

end even_natural_number_factors_count_l372_372645


namespace second_player_cannot_win_l372_372085

-- Definition of a tic-tac-toe board and optimal play conditions
def ttt_board := fin 3 × fin 3
def player := ℕ
def X : player := 1
def O : player := 2

-- The game state definition
structure game_state :=
(current_board : ttt_board → option player)
(current_player : player)

-- Definition of optimal play by the first player
def optimal_play (s : game_state) : Prop :=
∀ s', s'.current_player = X → s'.current_board = s.current_board ↔ s'.current_board = some X

-- Proof statement that the second player cannot win with optimal play from the first player
theorem second_player_cannot_win (s : game_state) 
(optimal_X : optimal_play s) :
  ¬ ∃ b : ttt_board, b.current_board = some O → (b = (0, 0) ∨ b = (0, 2) ∨ b = (2, 0) ∨ b = (2, 2) ∨ b = (0, 1) ∨ b = (1, 0) ∨ b = (1, 2) ∨ b = (2, 1)) :=
sorry

end second_player_cannot_win_l372_372085


namespace percent_more_than_l372_372329

-- Definitions and conditions
variables (x y p : ℝ)

-- Condition: x is p percent more than y
def x_is_p_percent_more_than_y (x y p : ℝ) : Prop :=
  x = y + (p / 100) * y

-- The theorem to prove
theorem percent_more_than (h : x_is_p_percent_more_than_y x y p) :
  p = 100 * (x / y - 1) :=
sorry

end percent_more_than_l372_372329


namespace quadrilateral_choices_l372_372416

theorem quadrilateral_choices :
  let available_rods : List ℕ := (List.range' 1 41).diff [5, 12, 20]
  let valid_rods := available_rods.filter (λ x => 4 ≤ x ∧ x ≤ 36)
  valid_rods.length = 30 := sorry

end quadrilateral_choices_l372_372416


namespace solve_for_a_l372_372006

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372006


namespace last_number_written_on_paper_l372_372941

-- Define the conditions of the problem and the final goal.

def paper_width : ℕ := 100
def paper_height : ℕ := 100
def total_characters : ℕ := paper_width * paper_height

theorem last_number_written_on_paper : 
  (∀ (w h : ℕ), w = paper_width ∧ h = paper_height →
   (∃ last_number : ℕ, 
    (∃ (f : ℕ → ℕ), 
      (∀ n >= 1, f(n) = n) ∧ -- the sequence function
      (sum (λ n, string.length (to_string (f n)) + 1) 1 last_number) <= total_characters ∧ 
      (sum (λ n, string.length (to_string (f n)) + 1) 1 (last_number + 1)) > total_characters))) →
  last_number = 2802 :=
sorry

end last_number_written_on_paper_l372_372941


namespace find_integer_l372_372547

theorem find_integer (x : ℕ) (h : (4 * x) ^ 2 - 3 * x = 1764) : x = 18 := 
by 
  sorry

end find_integer_l372_372547


namespace not_power_of_two_l372_372478

theorem not_power_of_two (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ∀ k : ℤ, (36 * a + b) * (a + 36 * b) ≠ 2 ^ k := by
  sorry

end not_power_of_two_l372_372478


namespace max_binary_sequences_len24_l372_372847

/--
  The maximum number of binary sequences of length 24,
  such that any two sequences differ in at least 8 positions,
  is at most 4096.
-/
theorem max_binary_sequences_len24 :
  ∀ (S : set (vector bool 24)), 
    (∀ (x y ∈ S), x ≠ y → dist x y ≥ 8) → 
    S.finite ∧ S.card ≤ 4096 :=
by
  sorry

end max_binary_sequences_len24_l372_372847


namespace proof_statements_imply_negation_l372_372992

-- Define propositions p, q, and r
variables (p q r : Prop)

-- Statement (1): p, q, and r are all true.
def statement_1 : Prop := p ∧ q ∧ r

-- Statement (2): p is true, q is false, and r is true.
def statement_2 : Prop := p ∧ ¬ q ∧ r

-- Statement (3): p is false, q is true, and r is false.
def statement_3 : Prop := ¬ p ∧ q ∧ ¬ r

-- Statement (4): p and r are false, q is true.
def statement_4 : Prop := ¬ p ∧ q ∧ ¬ r

-- The negation of "p and q are true, and r is false" is "¬(p ∧ q) ∨ r"
def negation : Prop := ¬(p ∧ q) ∨ r

-- Proof statement that each of the 4 statements implies the negation
theorem proof_statements_imply_negation :
  (statement_1 p q r → negation p q r) ∧
  (statement_2 p q r → negation p q r) ∧
  (statement_3 p q r → negation p q r) ∧
  (statement_4 p q r → negation p q r) :=
by
  sorry

end proof_statements_imply_negation_l372_372992


namespace pseudocode_final_value_Z_l372_372338

def execute_pseudocode (X Y Z : ℕ) : ℕ :=
  let rec loop : ℕ × ℕ × ℕ → ℕ
  | (X, Y, Z) =>
    if X < 10 then
      let Z' := Z * Y
      let X' := X + Z'
      loop (X', Y, Z')
    else
      Z
  loop (X, Y, Z)

theorem pseudocode_final_value_Z :
  execute_pseudocode 2 3 1 = 27 := sorry

end pseudocode_final_value_Z_l372_372338


namespace find_largest_integer_leq_inv_A_l372_372424

noncomputable def binom (n k : ℕ) : ℚ := n.choose k

def A : ℚ :=
  limit (λ n, ∑ i in range 2017, (-1) ^ i * (binom n i * binom n (i + 2)) / (binom n (i + 1)) ^ 2)

theorem find_largest_integer_leq_inv_A :
  (floor (1 / A) : ℤ) = 3 := by
  sorry

end find_largest_integer_leq_inv_A_l372_372424


namespace correct_calculation_option_B_l372_372548

theorem correct_calculation_option_B (a b : ℝ) :
  (-a^3 * b^5)^2 = a^6 * b^{10} :=
by sorry

end correct_calculation_option_B_l372_372548


namespace find_N_l372_372556

variable (N : ℝ)

def correctAnswer := (5 / 16) * N
def studentsAnswer := (5 / 6) * N

theorem find_N (h : studentsAnswer = correctAnswer + 200) : N = 384 := by
  sorry

end find_N_l372_372556


namespace number_1991_appears_in_199th_row_and_2nd_position_l372_372076

-- Definitions based on conditions
def is_multiple_of_ten (n : ℕ) : Prop := n % 10 = 0

def row_starts_with_multiple_of_ten (n : ℕ) : ℕ := (n / 10) * 10

def position_in_row (n : ℕ) : ℕ := match n % 10 with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  -- Add more if necessary
  | _ => 0 -- Default value for non-sequence numbers

-- The proof statement
theorem number_1991_appears_in_199th_row_and_2nd_position :
  ∃ n m k : ℕ,
  row_starts_with_multiple_of_ten 1991 = 1990 ∧
  position_in_row 1991 = 2 ∧
  1991 = row_starts_with_multiple_of_ten 1991 + m ∧
  (number_of_elements_up_to_row 199 k = 1991) :=
by
  sorry

end number_1991_appears_in_199th_row_and_2nd_position_l372_372076


namespace rectangular_curve_eq_general_form_eq_intersect_distance_l372_372792

-- Definitions of conditions
def polar_to_rectangular (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta in
  let y := rho * Real.sin theta in
  rho^2 = x^2 + y^2 ∧ rho * Real.cos theta = x

def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + (Real.sqrt 3 / 2) * t ∧ y = (1/2) * t

-- Proof goals given the conditions
theorem rectangular_curve_eq (rho theta x y : ℝ) (h1 : polar_to_rectangular rho theta) :
  (x - 2)^2 + y^2 = 4 :=
sorry

theorem general_form_eq (t x y : ℝ) (h2 : parametric_line t x y) :
  x - (Real.sqrt 3) * y - 3 = 0 :=
sorry

theorem intersect_distance (t1 t2 : ℝ) (Q : ℝ × ℝ) (hQ : Q = (3, 0)) (h1 : t1 + t2 = -Real.sqrt 3) (h2 : t1 * t2 = -3) :
  ||P_1Q|-|P_2Q|| = Real.sqrt 3 :=
sorry

end rectangular_curve_eq_general_form_eq_intersect_distance_l372_372792


namespace combined_tax_rate_l372_372610

theorem combined_tax_rate (M : ℝ) (h_positive : M > 0) :
  let Mork_tax_rate := 0.45 in
  let Mindy_tax_rate := 0.25 in
  let Mindy_income_multiplier := 4 in
  let Combined_income := M + 4 * M in
  let Combined_tax := Mork_tax_rate * M + Mindy_tax_rate * (4 * M) in
  (Combined_tax / Combined_income) * 100 = 29 :=
by
  sorry

end combined_tax_rate_l372_372610


namespace max_abs_x_y_l372_372372

theorem max_abs_x_y (x y : ℝ) (h : 4 * x^2 + y^2 = 4) : |x| + |y| ≤ 2 :=
by sorry

end max_abs_x_y_l372_372372


namespace triple_angle_cosine_identity_prove_cosine_identity_for_n_eq_4_l372_372993

open Real 

theorem triple_angle_cosine_identity (theta : ℝ) (n : ℕ) :
  (∀ k < n, cos (3^k * theta) ≠ 0) →
  ∏ k in Finset.range n, (4 * (cos (3^k * theta))^2 - 3) = (cos (3^n * theta) / cos theta) :=
sorry

theorem prove_cosine_identity_for_n_eq_4 (theta : ℝ) :
  (theta = 9 * π / 180) →
  (cos(81 * π / 180) = cos(9 * π / 180)) →
  ∏ k in Finset.range 4, (4 * (cos (3^k * theta))^2 - 3) = 1 :=
sorry

end triple_angle_cosine_identity_prove_cosine_identity_for_n_eq_4_l372_372993


namespace correct_statement_is_A_l372_372551

-- Define what a pyramid is
def is_pyramid (solid : Type) : Prop :=
  ∃ (face : solid → Prop) (polygon : solid → Prop) (triangles : solid → Prop),
  (∃ v, ∃ (adj : face → triangles), face polygon) ∧
  ∀ t : triangles, ∃ p, t θ (triangles p ∧ triangles v)

-- Define what a prism is
def is_prism (solid : Type) : Prop :=
  ∃ (face : solid → Prop) (parallel : solid → solid → Prop) (parallelograms : solid → Prop),
  (∃ f1 f2, parallel f1 f2) ∧
  (∀ e, parallelograms e ∧ (∃ v, e θ parallel))

-- lemmas to check the correctness of definitions
lemma condition_A_correct (solid : Type) : is_pyramid solid :=
sorry

lemma condition_B_correct (solid : Type) : is_prism solid :=
sorry

lemma condition_C_correct (solid : Type) : ¬ (is_pyramid solid) :=
sorry

lemma condition_D_correct (solid : Type) : ¬ (is_prism solid) :=
sorry

-- Main theorem
theorem correct_statement_is_A (solid : Type) :
  is_pyramid solid ∧ ¬ (is_prism solid) :=
by
  apply_and.intro
  { apply condition_A_correct }
  { apply and.intro
    { apply condition_C_correct }
    { apply condition_D_correct } }

end correct_statement_is_A_l372_372551


namespace smallest_number_of_eggs_proof_l372_372177

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l372_372177


namespace book_arrangement_l372_372937

theorem book_arrangement: 
  let A := 4  -- Number of Algebra Essentials
  let C := 5  -- Number of Calculus Fundamentals
  (∃! ways_to_arrange : ℕ, ways_to_arrange = 120 ↔ 
    ∃! CF_block: (list ℕ), 
      list.length CF_block = C ∧ 
      (∃! arr: (list (list ℕ)), 
        list.length arr = 1 + A ∧ 
        list.length (arr.erase CF_block) = A)) :=
by sorry

end book_arrangement_l372_372937


namespace sum_f_n_eq_26_l372_372679

noncomputable def f (n : ℕ) : ℝ :=
if (∃ m : ℕ, (Real.log n / Real.log 8) = m) then Real.log n / Real.log 8 else 0

theorem sum_f_n_eq_26 : (∑ n in Finset.range 4096, f n) = 26 := 
by
  sorry

end sum_f_n_eq_26_l372_372679


namespace ramesh_paid_price_l372_372087

-- Define the variables based on the conditions
variable (labelledPrice transportCost installationCost sellingPrice paidPrice : ℝ)

-- Define the specific values given in the problem
def discount : ℝ := 0.20 
def profitRate : ℝ := 0.10 
def actualSellingPrice : ℝ := 24475
def transportAmount : ℝ := 125
def installationAmount : ℝ := 250

-- Define the conditions given in the problem as Lean definitions
def selling_price_no_discount (P : ℝ) : ℝ := (1 + profitRate) * P
def discounted_price (P : ℝ) : ℝ := P * (1 - discount)
def total_cost (P : ℝ) : ℝ :=  discounted_price P + transportAmount + installationAmount

-- The problem is to prove that the price Ramesh paid for the refrigerator is Rs. 18175
theorem ramesh_paid_price : 
  ∀ (labelledPrice : ℝ), 
  selling_price_no_discount labelledPrice = actualSellingPrice → 
  paidPrice = total_cost labelledPrice → 
  paidPrice = 18175 := 
by
  intros labelledPrice h1 h2 
  sorry

end ramesh_paid_price_l372_372087


namespace ellipse_eccentricity_l372_372047

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372047


namespace range_of_f_l372_372677

noncomputable def f (x : ℝ) : ℝ := 
  (sin x ^ 3 + 6 * sin x ^ 2 + sin x + 2 * cos x ^ 2 - 8) / (sin x - 1)

theorem range_of_f : set.Ico 2 12 = {y | ∃ x ∈ ℝ, y = f x ∧ sin x ≠ 1} :=
sorry

end range_of_f_l372_372677


namespace congruent_and_orthocenter_of_reflected_triangle_l372_372446

theorem congruent_and_orthocenter_of_reflected_triangle
    {A B C O A_1 B_1 C_1 : Point}
    (hO : is_circumcenter O △ABC)
    (hA_1 : A_1 = reflect O BC)
    (hB_1 : B_1 = reflect O CA)
    (hC_1 : C_1 = reflect O AB) :
  (△A_1 B_1 C_1 ≅ △ABC) ∧ (is_orthocenter O △A_1 B_1 C_1) :=
by
  sorry

end congruent_and_orthocenter_of_reflected_triangle_l372_372446


namespace frosting_needed_correct_l372_372836

def frosting_needed (
  layer_cakes : ℕ,
  tiered_cakes : ℕ,
  dozen_cupcakes : ℕ,
  mini_cupcakes : ℕ,
  single_cakes : ℕ,
  pans_brownies : ℕ,
  frosting_per_layer_cake : ℚ,
  frosting_per_tiered_cake : ℚ,
  frosting_per_dozen_cupcakes : ℚ,
  frosting_per_pair_mini_cupcakes : ℚ,
  frosting_per_single_cake : ℚ,
  frosting_per_pan_brownies : ℚ
) : ℚ :=
  layer_cakes * frosting_per_layer_cake +
  tiered_cakes * frosting_per_tiered_cake +
  dozen_cupcakes * frosting_per_dozen_cupcakes +
  (mini_cupcakes / 2) * frosting_per_pair_mini_cupcakes +
  single_cakes * frosting_per_single_cake +
  pans_brownies * frosting_per_pan_brownies

theorem frosting_needed_correct :
  frosting_needed 4 8 10 30 15 24 1 1.5 0.5 0.25 0.5 0.5 = 44.25 :=
by
  sorry

end frosting_needed_correct_l372_372836


namespace youngest_child_age_l372_372582

theorem youngest_child_age (total_bill mother's_charge child_charge n_triplets : ℝ) 
  (triplet_age youngest_child_age : ℝ) : 
  total_bill = 18.75 →
  mother's_charge = 9.75 →
  child_charge = 0.75 →
  n_triplets = 3 →
  total_bill - mother's_charge = child_charge * (n_triplets * triplet_age + youngest_child_age) →
  (youngest_child_age = 3 ∨ youngest_child_age = 6) :=
begin
  intros h1 h2 h3 h4 h5,
  sorry,
end

end youngest_child_age_l372_372582


namespace find_a_l372_372018

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l372_372018


namespace circumcircle_through_midpoint_l372_372361

section

variables {O1 O2 : Point}
variables {l1 l2 l3 : Line} -- The three given lines
variables {circle1 circle2 : Circle} -- The two given circles
variables (midpoint_P : Point) -- Midpoint of the segment O1 O2

-- Define the property that each line intersects the circles at chords of equal length
def equal_chord_length (l : Line) (circle1 circle2 : Circle) : Prop :=
∃ (chord1 chord2 : Chord), chord1.length = chord2.length ∧
  chord1.line = l ∧ chord2.line = l ∧ chord1.circle = circle1 ∧ chord2.circle = circle2

-- Define the three lines each intersecting the circles at chords of equal length
axiom h1 : equal_chord_length l1 circle1 circle2
axiom h2 : equal_chord_length l2 circle1 circle2
axiom h3 : equal_chord_length l3 circle1 circle2

-- Define the intersection points of the lines forming a triangle
variables {A B C : Point} -- Intersection points of lines l1, l2, and l3 forming a triangle

-- The circumcircle of triangle passes through the midpoint of the segment O1 O2
theorem circumcircle_through_midpoint (circumcircle : Circle)
  (circumcircle_def : Circumcircle A B C circumcircle) :
  OnCircumcircle A B C midpoint_P :=
sorry

end

end circumcircle_through_midpoint_l372_372361


namespace faster_train_crosses_slower_l372_372902

def kmph_to_mps (speed_kmph : ℕ) : ℝ :=
  speed_kmph * 5 / 18

def relative_speed (speed1 speed2 : ℕ) : ℝ :=
  kmph_to_mps (speed1 - speed2)

theorem faster_train_crosses_slower :
  let S_fast := 108
      S_slow := 36
      L_fast := 340
      rel_speed := relative_speed S_fast S_slow in
  (L_fast : ℝ) / rel_speed = 17 :=
by
  sorry

end faster_train_crosses_slower_l372_372902


namespace second_largest_times_smallest_l372_372145

theorem second_largest_times_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) : b * a = 110 :=
by
  rw [h1, h2]
  sorry

end second_largest_times_smallest_l372_372145


namespace product_of_two_smaller_numbers_is_85_l372_372523

theorem product_of_two_smaller_numbers_is_85
  (A B C : ℝ)
  (h1 : B = 10)
  (h2 : C - B = B - A)
  (h3 : B * C = 115) :
  A * B = 85 :=
by
  sorry

end product_of_two_smaller_numbers_is_85_l372_372523


namespace largest_arithmetic_mean_of_two_digit_numbers_l372_372286

theorem largest_arithmetic_mean_of_two_digit_numbers 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100)
  (h3 : a > b)
  (h4 : (a + b : ℝ) / 2 = 25/24 * real.sqrt (a * b)) : (a + b) / 2 = 75 :=
sorry

end largest_arithmetic_mean_of_two_digit_numbers_l372_372286


namespace polynomial_value_at_2_l372_372905

def f (x : ℝ) : ℝ := 2 * x^5 + 4 * x^4 - 2 * x^3 - 3 * x^2 + x

theorem polynomial_value_at_2 : f 2 = 102 := by
  sorry

end polynomial_value_at_2_l372_372905


namespace product_of_solutions_product_of_all_t_l372_372671

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372671


namespace find_square_length_CD_l372_372083

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x - 2

def is_midpoint (mid C D : (ℝ × ℝ)) : Prop :=
  mid.1 = (C.1 + D.1) / 2 ∧ mid.2 = (C.2 + D.2) / 2

theorem find_square_length_CD (C D : ℝ × ℝ)
  (hC : C.2 = parabola C.1)
  (hD : D.2 = parabola D.1)
  (h_mid : is_midpoint (0,0) C D) :
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
sorry

end find_square_length_CD_l372_372083


namespace absolute_value_z_squared_minus_2z_l372_372756

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem
theorem absolute_value_z_squared_minus_2z : complex.abs (z^2 - 2*z) = 2 := by
  sorry

end absolute_value_z_squared_minus_2z_l372_372756


namespace matrix_det_correct_l372_372279

variable (x y z : ℝ)

theorem matrix_det_correct : 
  det ![
  ![2, x, y + z],
  ![2, x + y, y + z],
  ![2, x + z, x + y]
  ] = 5 * x * y + 4 * x * z - 6 * z^2 := by
    sorry

end matrix_det_correct_l372_372279


namespace remainder_12401_163_l372_372909

theorem remainder_12401_163 :
  let original_number := 12401
  let divisor := 163
  let quotient := 76
  let remainder := 13
  original_number = divisor * quotient + remainder :=
by
  sorry

end remainder_12401_163_l372_372909


namespace product_of_solutions_of_t_squared_eq_49_l372_372665

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372665


namespace product_of_solutions_of_t_squared_eq_49_l372_372670

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372670


namespace ellipse_eccentricity_a_l372_372012

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372012


namespace probability_wife_selection_l372_372939

theorem probability_wife_selection (P_H P_only_one P_W : ℝ)
  (h1 : P_H = 1 / 7)
  (h2 : P_only_one = 0.28571428571428575)
  (h3 : P_only_one = (P_H * (1 - P_W)) + (P_W * (1 - P_H))) :
  P_W = 1 / 5 :=
by
  sorry

end probability_wife_selection_l372_372939


namespace professional_doctors_percentage_l372_372250

-- Defining the context and conditions:

variable (total_percent : ℝ) (leaders_percent : ℝ) (nurses_percent : ℝ) (doctors_percent : ℝ)

-- Specifying the conditions:
def total_percentage_sum : Prop :=
  total_percent = 100

def leaders_percentage : Prop :=
  leaders_percent = 4

def nurses_percentage : Prop :=
  nurses_percent = 56

-- Stating the actual theorem to be proved:
theorem professional_doctors_percentage
  (h1 : total_percentage_sum total_percent)
  (h2 : leaders_percentage leaders_percent)
  (h3 : nurses_percentage nurses_percent) :
  doctors_percent = 100 - (leaders_percent + nurses_percent) := by
  sorry -- Proof placeholder

end professional_doctors_percentage_l372_372250


namespace abs_diff_of_roots_quadratic_l372_372626

noncomputable def roots_of_quadratic_eq (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b * b - 4 * a * c
  if discriminant < 0 then (0, 0) -- For simplicity assuming noncomplex solutions
  else ((-b + (Real.sqrt discriminant)) / (2 * a), (-b - (Real.sqrt discriminant)) / (2 * a))

noncomputable def abs_diff_of_roots (a b c : ℝ) : ℝ :=
  let (r1, r2) := roots_of_quadratic_eq a b c
  Real.abs (r1 - r2)

theorem abs_diff_of_roots_quadratic :
  abs_diff_of_roots 1 (-7) 10 = 3 :=
by
  -- The proof should verify this theorem, hence we add a sorry for now.
  sorry

end abs_diff_of_roots_quadratic_l372_372626


namespace total_weight_of_packages_l372_372520

theorem total_weight_of_packages
  (initial_ratio : ℕ × ℕ)
  (moved_grams : ℕ)
  (new_ratio : ℕ × ℕ)
  (initial_ratio_eq : initial_ratio = (4, 1))
  (moved_grams_eq : moved_grams = 10)
  (new_ratio_eq : new_ratio = (7, 8)) :
  let total_weight := moved_grams / ((4.to_rat / 5) - (7.to_rat / 15)) in
  total_weight = 30 :=
by
  sorry

end total_weight_of_packages_l372_372520


namespace range_values_for_a_l372_372321

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x a : ℝ) (ha : 0 < a) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0

theorem range_values_for_a (a : ℝ) : (∃ ha : 0 < a, (∀ x : ℝ, (¬ p x → q x a ha))) → (0 < a ∧ a ≤ 3) :=
by
  sorry

end range_values_for_a_l372_372321


namespace sin_squared_general_l372_372364

theorem sin_squared_general (α : ℝ) :
  (sin (α - (π / 3)))^2 + (sin α)^2 + (sin (α + (π / 3)))^2 = 3 / 2 :=
by
  -- translating the 2 conditions into Lean terms
  have h1 : (sin (π / 6))^2 + (sin (π / 2))^2 + (sin (5 * π / 6))^2 = 3 / 2 := sorry,
  have h2 : (sin (5 * π / 180))^2 + (sin (65 * π / 180))^2 + (sin (125 * π / 180))^2 = 3 / 2 := sorry,
  sorry

end sin_squared_general_l372_372364


namespace radius_equiv_l372_372931

-- Define the radius in its original form
def radius_original : ℝ := 0.000000005

-- Define the radius in scientific notation
def radius_scientific : ℝ := 5 * 10^(-9)

-- The theorem to prove equivalence between the two
theorem radius_equiv :
  radius_original = radius_scientific :=
sorry

end radius_equiv_l372_372931


namespace smallest_square_area_contains_rectangles_l372_372200

theorem smallest_square_area_contains_rectangles :
  ∀ x y a b : ℝ, (x = 2) ∧ (y = 3) ∧ (a = 4) ∧ (b = 5) →
  ∃ L : ℝ, (L >= max (x + a) (y + b)) ∧ (L ^ 2 = 49) :=
by
  intros x y a b h
  cases h with hx rest
  cases rest with hy rest'
  cases rest' with ha hb
  sorry

end smallest_square_area_contains_rectangles_l372_372200


namespace product_of_t_values_l372_372659

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372659


namespace find_fraction_l372_372567

theorem find_fraction (x f : ℝ) (h₁ : x = 140) (h₂ : 0.65 * x = f * x - 21) : f = 0.8 :=
by
  sorry

end find_fraction_l372_372567


namespace equilateral_triangle_sum_l372_372166

theorem equilateral_triangle_sum (side_length : ℚ) (h_eq : side_length = 13 / 12) :
  3 * side_length = 13 / 4 :=
by
  -- Proof omitted
  sorry

end equilateral_triangle_sum_l372_372166


namespace consecutive_odd_integers_l372_372565

theorem consecutive_odd_integers (x : ℤ) (h : x + 4 = 15) : 3 * x - 2 * (x + 4) = 3 :=
by
  sorry

end consecutive_odd_integers_l372_372565


namespace calc_a5_l372_372996

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sequence_a : ℕ → ℕ
| 1     := 1989 ^ 1989
| (n+1) := sum_of_digits (sequence_a n)

theorem calc_a5 : sequence_a 5 = 9 := 
by {
  -- Proof can be filled here
  sorry
}

end calc_a5_l372_372996


namespace det_trig_matrix_eq_one_l372_372277

variable {α β : ℝ}

theorem det_trig_matrix_eq_one (α β : ℝ) : 
  det (λ i j, ![![cos α * cos β, cos α * sin β, -sin α], 
                 ![-sin β, cos β, 0],
                 ![sin α * cos β, sin α * sin β, cos α]] i j) = 1 :=
by
  sorry

end det_trig_matrix_eq_one_l372_372277


namespace smallest_integer_geq_l372_372540

theorem smallest_integer_geq : ∃ (n : ℤ), (n^2 - 9*n + 18 ≥ 0) ∧ ∀ (m : ℤ), (m^2 - 9*m + 18 ≥ 0) → n ≤ m :=
by
  sorry

end smallest_integer_geq_l372_372540


namespace find_primes_l372_372642

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end find_primes_l372_372642


namespace abs_z_squared_minus_two_z_eq_two_l372_372745

theorem abs_z_squared_minus_two_z_eq_two (z : ℂ) (hz : z = 1 + 1*Complex.i) : 
  |z^2 - 2*z| = 2 :=
begin
  sorry
end

end abs_z_squared_minus_two_z_eq_two_l372_372745


namespace no_such_f_cos_2x_l372_372892

theorem no_such_f_cos_2x (f : ℝ → ℝ) : ¬ (∀ x : ℝ, f (cos (2 * x)) = cos x) := 
sorry

end no_such_f_cos_2x_l372_372892


namespace system_solutions_l372_372482

theorem system_solutions (x y z a b c : ℝ) :
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0) → (¬(x = 1 ∨ y = 1 ∨ z = 1) → 
  ∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ a + b + c + a * b * c ≠ 0) → 
  ¬∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c :=
by
    sorry

end system_solutions_l372_372482


namespace AM_perp_plane_A_dihedral_angle_A_A_l372_372782

noncomputable def AC : ℝ := 2
noncomputable def AA' : ℝ := 2
noncomputable def angle_A'AC : ℝ := 120 -- in degrees
noncomputable def angle_BA'C : ℝ := 30  -- in degrees
noncomputable def A'C := Real.sqrt (2^2 + 2^2 - 2 * 2 * 2 * Real.cos (120 * Real.pi / 180)) -- A'C using the cosine rule

noncomputable def M : ℝ := A'C / 2 -- midpoint

theorem AM_perp_plane_A'BC : 
AM ⊥ Plane (Points.A', Points.B, Points.C) := sorry

theorem dihedral_angle_A_A'BC :
dihedral_angle (A, Plane (Points.A', Points.B, Points.C)) = Real.atan 2 := sorry

end AM_perp_plane_A_dihedral_angle_A_A_l372_372782


namespace multiplication_table_odd_fraction_l372_372972

def count_odd_products (n : ℕ) : ℕ :=
  let odd_count := Nat.card (Finset.filter (fun x => x % 2 = 1) (Finset.range (n + 1)))
  in odd_count * odd_count

theorem multiplication_table_odd_fraction :
  (count_odd_products 15 : ℚ) / 256 = 1 / 4 :=
by
  -- The proof is skipped as per the requirements
  sorry

end multiplication_table_odd_fraction_l372_372972


namespace growth_rate_equation_l372_372911

variable (a x : ℝ)

-- Condition: The number of visitors in March is three times that of January
def visitors_in_march := 3 * a

-- Condition: The average growth rate of visitors in February and March is x
def growth_rate := x

-- Statement to prove
theorem growth_rate_equation 
  (h : (1 + x)^2 = 3) : true :=
by sorry

end growth_rate_equation_l372_372911


namespace find_a_l372_372350

-- Define the function f
def f (a x : ℝ) : ℝ :=
  a * x - (3 / 2) * x ^ 2

-- Conditions
def cond1 (a : ℝ) : Prop :=
  ∀ x, f a x ≤ 1 / 6

def cond2 (a : ℝ) : Prop :=
  ∀ x, (1 / 4 ≤ x ∧ x ≤ 1 / 2) → f a x ≥ 1 / 8

-- Statement to prove
theorem find_a (a : ℝ) (h1 : cond1 a) (h2 : cond2 a) : a = 1 := by
  sorry

end find_a_l372_372350


namespace sin_cubic_trig_identity_sum_l372_372295

open Real

noncomputable def sum_of_x_values (x : ℝ) : ℝ :=
if (30 < x ∧ x < 90 ∧ sin (2 * x) ^ 3 + sin (6 * x) ^ 3 = 8 * (sin (3 * x) ^ 3) * (sin x ^ 3)) then x else 0

theorem sin_cubic_trig_identity_sum :
  ∑ x in {30 < x ∧ x < 90 ∧ sin (2 * x) ^ 3 + sin (6 * x) ^ 3 = 8 * (sin (3 * x) ^ 3) * (sin x ^ 3)}, sum_of_x_values x = 135 :=
sorry

end sin_cubic_trig_identity_sum_l372_372295


namespace arithmetic_mean_fraction_l372_372339

theorem arithmetic_mean_fraction (a b c : ℚ) (h1 : a = 8 / 11) (h2 : b = 7 / 11) (h3 : c = 9 / 11) :
  a = (b + c) / 2 :=
by
  sorry

end arithmetic_mean_fraction_l372_372339


namespace bee_honeycomb_path_l372_372833

theorem bee_honeycomb_path (x1 x2 x3 : ℕ) (honeycomb_grid : Prop)
  (shortest_path : ℕ) (honeycomb_property : shortest_path = 100)
  (path_decomposition : x1 + x2 + x3 = 100) : x1 = 50 ∧ x2 + x3 = 50 := 
sorry

end bee_honeycomb_path_l372_372833


namespace ellipse_eccentricity_a_l372_372011

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372011


namespace find_digits_sum_l372_372788

theorem find_digits_sum (a b c : Nat) (ha : 0 <= a ∧ a <= 9) (hb : 0 <= b ∧ b <= 9) 
  (hc : 0 <= c ∧ c <= 9) 
  (h1 : 2 * a = c) 
  (h2 : b = b) : 
  a + b + c = 11 :=
  sorry

end find_digits_sum_l372_372788


namespace factorization_l372_372638

variable (a x : ℝ)

theorem factorization : a * x ^ 3 + x + a + 1 = (x + 1) * (a * x ^ 2 - a * x + a + 1) := 
sorry

end factorization_l372_372638


namespace A_worked_days_l372_372928

noncomputable def A_work_rate (W : ℕ) : ℝ := W / 15
noncomputable def B_work_rate (W : ℕ) : ℝ := W / 4.5
constant W : ℕ -- Denote the total work by W

-- B works for 3 days to complete the remaining work
constant B_days : ℝ := 3

-- Proving that A worked for 5 days before leaving the job
theorem A_worked_days : ∃ (x : ℝ), x = 5 ∧
  (x * A_work_rate W + B_days * B_work_rate W = W) :=
begin
  use 5,
  split,
  { refl },
  { sorry }
end

end A_worked_days_l372_372928


namespace robert_reading_books_l372_372473

theorem robert_reading_books (v t p : ℕ) (v_eq : v = 120) (t_eq : t = 8) (p_eq : p = 360) :
  let pages_read := v * t,
      full_books := pages_read / p in
  full_books = 2 := 
by 
  sorry

end robert_reading_books_l372_372473


namespace similar_triangle_perimeter_l372_372965

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∨ T.a = T.c ∨ T.b = T.c

def similar_triangles (T1 T2 : Triangle) : Prop :=
  T1.a / T2.a = T1.b / T2.b ∧ T1.b / T2.b = T1.c / T2.c ∧ T1.a / T2.a = T1.c / T2.c

noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

theorem similar_triangle_perimeter
  (T1 T2 : Triangle)
  (T1_isosceles : is_isosceles T1)
  (T1_sides : T1.a = 7 ∧ T1.b = 7 ∧ T1.c = 12)
  (T2_similar : similar_triangles T1 T2)
  (T2_longest_side : T2.c = 30) :
  perimeter T2 = 65 :=
by
  sorry

end similar_triangle_perimeter_l372_372965


namespace integral_value_l372_372636

open IntervalIntegral Real

noncomputable def integral_equiv : ℝ :=
  ∫ x in 0..1, sqrt(1 - (x - 1)^2)

theorem integral_value : integral_equiv = π / 4 := by
  sorry

end integral_value_l372_372636


namespace sum_of_cubes_of_roots_l372_372125

theorem sum_of_cubes_of_roots :
  ∀ (x1 x2 : ℝ), (2 * x1^2 - 5 * x1 + 1 = 0) ∧ (2 * x2^2 - 5 * x2 + 1 = 0) →
  (x1 + x2 = 5 / 2) ∧ (x1 * x2 = 1 / 2) →
  (x1^3 + x2^3 = 95 / 8) :=
by
  sorry

end sum_of_cubes_of_roots_l372_372125


namespace correct_equation_l372_372549

theorem correct_equation (a b c m : ℝ) (ha_ne_hb : a ≠ b) :
    (∃ (d : ℕ), 
        (d = 4 ∧ 
            (¬(abs b / abs a = abs (b^2) / abs (a^2))) ∧ 
            (¬(abs (-m^2 - 2) / abs (-m + 1) = abs (m^2 - 2) / abs (m + 1))) ∧ 
            (¬(abs b / abs a = abs (b + c) / abs (a + c))) ∧ 
            ((abs (a^2 - b^2) / abs ((a - b)^2)) = (abs (a + b) / abs (a - b)))))
 :=
⟨4, by {
    split;
    sorry  -- Provide the necessary argument here
}⟩

end correct_equation_l372_372549


namespace faster_train_pass_time_l372_372903

-- Defining the conditions
def length_of_train : ℕ := 45 -- length in meters
def speed_of_faster_train : ℕ := 45 -- speed in km/hr
def speed_of_slower_train : ℕ := 36 -- speed in km/hr

-- Define relative speed
def relative_speed := (speed_of_faster_train - speed_of_slower_train) * 5 / 18 -- converting km/hr to m/s

-- Total distance to pass (sum of lengths of both trains)
def total_passing_distance := (2 * length_of_train) -- 2 trains of 45 meters each

-- Calculate the time to pass the slower train
def time_to_pass := total_passing_distance / relative_speed

-- The theorem to prove
theorem faster_train_pass_time : time_to_pass = 36 := by
  -- This is where the proof would be placed
  sorry

end faster_train_pass_time_l372_372903


namespace product_of_solutions_product_of_all_t_l372_372676

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372676


namespace corn_syrup_amount_l372_372405

-- Definitions based on given conditions
def flavoring_to_corn_syrup_standard := 1 / 12
def flavoring_to_water_standard := 1 / 30

def flavoring_to_corn_syrup_sport := (3 * flavoring_to_corn_syrup_standard)
def flavoring_to_water_sport := (1 / 2) * flavoring_to_water_standard

def common_factor := (30 : ℝ)

-- Amounts in sport formulation after adjustment
def flavoring_to_corn_syrup_ratio_sport := 1 / 4
def flavoring_to_water_ratio_sport := 1 / 60

def total_flavoring_corn_syrup := 15 -- Since ratio is 15:60:60 and given water is 15 ounces

theorem corn_syrup_amount (water_ounces : ℝ) :
  water_ounces = 15 → 
  (60 / 60) * water_ounces = 15 :=
by
  sorry

end corn_syrup_amount_l372_372405


namespace max_elements_l372_372822

variable {α : Type*}
def satisfies_property (M : finset α) [LinearOrderedAddCommGroup α] : Prop :=
  ∀ (a b c ∈ M), a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M

theorem max_elements (M : finset α) [LinearOrderedAddCommGroup α] (hM : satisfies_property M) :
  M.card ≤ 7 :=
sorry

end max_elements_l372_372822


namespace triangle_similarity_l372_372412

variables {A B C H M K : Type*} [triangle A B C]
variables (altitude : is_altitude B H A C) (median1 : is_median A M B C) (median2 : is_median C K A B)
variables (midline : is_midline M K A C)

theorem triangle_similarity (hMK_parallel_AC : MK ∥ AC) (hMK_AC : MK = 1/2 * AC)
                            (hHM_bisect_BC : is_median B H C)
                            (hHM_half_BC : HM = 1/2 * BC) :
  similar (triangle K H M) (triangle A B C) :=
sorry

end triangle_similarity_l372_372412


namespace percentage_increase_salary_l372_372370

theorem percentage_increase_salary (S : ℝ) (P : ℝ) (h1 : 1.16 * S = 348) (h2 : S + P * S = 375) : P = 0.25 :=
by
  sorry

end percentage_increase_salary_l372_372370


namespace probability_interval_l372_372510

theorem probability_interval (P_A P_B P_A_inter_P_B : ℝ) (h1 : P_A = 3 / 4) (h2 : P_B = 2 / 3) : 
  5/12 ≤ P_A_inter_P_B ∧ P_A_inter_P_B ≤ 2/3 :=
sorry

end probability_interval_l372_372510


namespace g_ten_l372_372118

variable {g : ℝ → ℝ}

-- Define the conditions
axiom functional_equation (x y : ℝ) : g(x + y) = g(x) + g(y)
axiom g_three : g(3) = 4

-- Statement of the theorem to prove
theorem g_ten : g(10) = 40 / 3 := by
  sorry

end g_ten_l372_372118


namespace continuity_of_f_l372_372313

noncomputable theory

variables {R : Type*} [TopologicalSpace R] [OrderedAddCommGroup R] [TopologicalAddGroup R] 

-- Define f as a function from R to R
variable (f : R → R)

-- Assume the given conditions
axioms (h : ∀ a > (1 : R), Continuous (λ x, f x + f (a * x)))

-- Prove that f is continuous
theorem continuity_of_f : Continuous f :=
sorry

end continuity_of_f_l372_372313


namespace ratio_area_tris_XYD_XZD_altitude_from_D_to_YZ_l372_372408

noncomputable def TriangleXYZ := Type
variables (X Y Z D : TriangleXYZ)
variables (XY XZ YZ : ℝ)
variables (H1 : XY = 18)
variables (H2 : XZ = 27)
variables (H3 : YZ = 30)
variables (H4 : ∀ (P : TriangleXYZ), P = D → Angle(X, Y, P) = Angle(X, P, Z))

theorem ratio_area_tris_XYD_XZD :
  (area (Triangle.mk X Y D) / area (Triangle.mk X Z D)) = 2/3 := by
sorry

theorem altitude_from_D_to_YZ : 
  length_altitude(D, YZ) = 16.237 := by
sorry

end ratio_area_tris_XYD_XZD_altitude_from_D_to_YZ_l372_372408


namespace rearrange_5056_l372_372731

theorem rearrange_5056 : (finset.univ.filter (λ n : ℕ, 
                                       set.to_finset (nat.digits 10 n) = {5, 0, 5, 6} ∧ 
                                       1000 ≤ n ∧ n < 10000 ∧ 
                                       nat.digits 10 n !! 0 ≠ 0)).card = 9 := 
sorry

end rearrange_5056_l372_372731


namespace roja_speed_l372_372843

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end roja_speed_l372_372843


namespace combined_dilation_rotation_is_correct_l372_372292

noncomputable def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], ![0, 2]]

noncomputable def rotation_matrix_45 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2], ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]]

noncomputable def combined_transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix ⬝ rotation_matrix_45

theorem combined_dilation_rotation_is_correct :
  combined_transformation_matrix = ![![Real.sqrt 2, -Real.sqrt 2], ![Real.sqrt 2, Real.sqrt 2]] :=
by
  sorry

end combined_dilation_rotation_is_correct_l372_372292


namespace increasing_condition_maximum_value_l372_372447

noncomputable def f (a x : ℝ) := x - a * real.sqrt (x ^ 2 + 1) + a

theorem increasing_condition {a : ℝ} (h₀ : 0 < a) (h₁ : a ≤ real.sqrt 2) :
  ∀ x ∈ set.Icc (0 : ℝ) 1, f a x ≤ f a 1 := sorry

theorem maximum_value {a : ℝ}
  (h₀ : 0 < a) :
  (if h₁ : a ≤ real.sqrt 2 then
    ∃ x ∈ Icc (0 : ℝ) 1, f a x = 1 + a * (1 - real.sqrt 2)
  else
    ∃ x ∈ Icc (0 : ℝ) 1, f a x = a - real.sqrt (a ^ 2 - 1)) := sorry

end increasing_condition_maximum_value_l372_372447


namespace magnitude_of_z_l372_372702

def z := (1 + 2 * Complex.i) / (2 - Complex.i)

theorem magnitude_of_z : Complex.abs z = 1 :=
by
  sorry

end magnitude_of_z_l372_372702


namespace product_of_roots_eq_negative_forty_nine_l372_372656

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l372_372656


namespace bc_geq_b_plus_c_l372_372853

theorem bc_geq_b_plus_c {a b c : ℝ} (h : a + b + c = 2 * real.sqrt (a * b * c)) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : b * c ≥ b + c :=
sorry

end bc_geq_b_plus_c_l372_372853


namespace unique_solutions_count_l372_372283

theorem unique_solutions_count :
  ∃ (count : ℕ), count = 8 ∧ 
  ∃ (A B C : ℕ), (B ≠ 1 ∧ C ≠ 1 ∧ A ≠ 1) ∧ ([A * (10 * 1 + B + C)]^2 = 9025) ∧ 
                 ∃ (candidates : List (ℕ × ℕ × ℕ)), 
                   candidates = List.filter (λ (t : ℕ × ℕ × ℕ), 
                                             match t with
                                             | (a, b, c) => (b ≠ 1 ∧ c ≠ 1 ∧ a ≠ 1) ∧ ([a * (10 * 1 + b + c)]^2 = 9025)
                                             end)
                                             [(5,1,18), (5,2,17), (5,3,16), (5,4,15), (5,5,14), (5,6,13), (5,7,12), (5,8,11)]
                  ∧ List.length candidates = count :=
begin
  sorry
end

end unique_solutions_count_l372_372283


namespace find_k_of_period_l372_372769

theorem find_k_of_period:
  ∀ k : ℝ, (0 < k) ∧ (∀ x : ℝ, cos (k * x + (π / 6)) = cos (k * (x + 4 * π) + (π / 6))) → k = (1 / 2) :=
by
  intro k
  intro h
  sorry

end find_k_of_period_l372_372769


namespace arithmetic_sequence_properties_l372_372335

theorem arithmetic_sequence_properties
  (a₃ a₆ : ℤ) (h₃ : a₃ = 8) (h₆ : a₆ = 17)
  (a_n d b_n S_n : ℕ → ℤ)
  (h₁ : ∀ n, a_n = a_n 1 + (n-1) * d)
  (h₂ : ∀ n, a₁ = 2)
  (h₃' : ∀ n, d = 3)
  (h₄ : ∀ n, a_3 = a₁ + 2 * d)
  (h₅ : ∀ n, a_6 = a₁ + 5 * d)
  (h₆ : ∀ n, b_n = a_n + 2^(n-1))
  (h₇ : ∀ n, S_n = (∑ i in finset.range n, b_i) :
  (a₁ = 2) ∧ (d = 3) ∧ (S_n = (3 * n * n + n) / 2 + 2^n - 1) := 
sorry

end arithmetic_sequence_properties_l372_372335


namespace ellipse_eccentricity_a_l372_372010

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372010


namespace ellipse_a_value_l372_372031

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372031


namespace abs_z_squared_minus_two_z_eq_two_l372_372744

theorem abs_z_squared_minus_two_z_eq_two (z : ℂ) (hz : z = 1 + 1*Complex.i) : 
  |z^2 - 2*z| = 2 :=
begin
  sorry
end

end abs_z_squared_minus_two_z_eq_two_l372_372744


namespace incorrect_statement_about_transformed_function_g_l372_372529

noncomputable def f (x : ℝ) : ℝ := √2 * Math.sin (2 * x) - √2 * Math.cos (2 * x) + 1
noncomputable def g (x : ℝ) : ℝ := 2 * Math.sin (2 * x + (Real.pi / 4))

-- Conditions and required statements for verification:
def smallest_positive_period_of_g_is_pi : Prop :=
  ∃ T > 0, ∀ x, g(x + T) = g x ∧ T = Real.pi

def symmetry_axis_of_g_is_pi_over_8 : Prop :=
  ∃ k : Int, ∃ x = k * (Real.pi / 2) + (Real.pi / 8), g x = g (-x)

def integral_of_g_over_interval_eq_sqrt_2 : Prop :=
  (∫ x in 0..(Real.pi / 2), g x) = √2

def g_is_not_monotonically_decreasing : Prop :=
  ∃ a b : ℝ, a = Real.pi / 12 ∧ b = 5 * Real.pi / 8 ∧ ¬ (∀ x y ∈ Icc a b, x ≤ y → g x ≥ g y)

theorem incorrect_statement_about_transformed_function_g :
  smallest_positive_period_of_g_is_pi ∧
  symmetry_axis_of_g_is_pi_over_8 ∧
  integral_of_g_over_interval_eq_sqrt_2 ∧
  g_is_not_monotonically_decreasing := sorry

end incorrect_statement_about_transformed_function_g_l372_372529


namespace product_formula_l372_372974

theorem product_formula :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) *
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) *
  (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end product_formula_l372_372974


namespace general_formula_a_n_sum_T_n_l372_372692

-- Define the sequence {a_n} and the sum S_n of its first n terms
def a (n : ℕ) : ℕ := 2^(n + 1)
def S (n : ℕ) : ℕ := 2 * a n - 4

-- Define the sequence {b_n}
def b (n : ℕ) : ℚ := (a n : ℚ) / (n * (n + 1) * 2^n)

-- Define the sum T_n of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b (k + 1))

-- Prove the general formula for the term a_n of the sequence {a_n}
theorem general_formula_a_n : ∀ n, S n = 2 * a n - 4 → a n = 2^(n + 1) := 
by 
  intros n h,
  induction n with d hd,
  { simp [a, S] at h,
    sorry },
  { simp [a, S] at h,
    sorry }

-- Prove the sum of the first n terms T_n of the sequence {b_n}
theorem sum_T_n : ∀ n, (T n = (2 * n) / (n + 1) : ℚ) := 
by 
  intros n,
  induction n with d hd,
  { simp [T, b, a] },
  { simp [T, b, a],
    sorry }

end general_formula_a_n_sum_T_n_l372_372692


namespace valid_pairs_l372_372639

theorem valid_pairs
  (x y : ℕ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_div : ∃ k : ℕ, k > 0 ∧ k * (2 * x + 7 * y) = 7 * x + 2 * y) :
  ∃ a : ℕ, a > 0 ∧ (x = a ∧ y = a ∨ x = 4 * a ∧ y = a ∨ x = 19 * a ∧ y = a) :=
by
  sorry

end valid_pairs_l372_372639


namespace imaginary_part_of_z_l372_372070

-- Define the complex number z
def z : ℂ :=
  3 - 2 * Complex.I

-- Lean theorem statement to prove the imaginary part of z is -2
theorem imaginary_part_of_z :
  Complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l372_372070


namespace boy_arrival_early_minutes_l372_372202

noncomputable def time_late_minutes := 7
noncomputable def distance := 3
noncomputable def speed_day1 := 6
noncomputable def speed_day2 := 12

theorem boy_arrival_early_minutes:
  let late_hours := (time_late_minutes : ℝ) / 60,
      time_day1 := distance / speed_day1,
      actual_time := time_day1 - late_hours,
      time_day2 := distance / speed_day2,
      early_hours := actual_time - time_day2 in
  early_hours * 60 = 8 :=
by
  sorry

end boy_arrival_early_minutes_l372_372202


namespace cos_pi_over_7_identity_l372_372470

theorem cos_pi_over_7_identity :
  cos (π / 7) - cos (2 * π / 7) + cos (3 * π / 7) = 1 / 2 :=
by
  sorry

end cos_pi_over_7_identity_l372_372470


namespace find_a_l372_372053

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372053


namespace smallest_four_digit_number_divisible_by_8_with_specific_digit_properties_l372_372165

theorem smallest_four_digit_number_divisible_by_8_with_specific_digit_properties :
  ∃ n : ℕ, (n ≥ 1000) ∧ (n < 10000) ∧ (8 ∣ n) ∧
    (∃ a b c d : ℕ, n = a * 1000 + b * 100 + c * 10 + d ∧
      (a ∈ {2}) ∧ ((b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) ∧ (bit1 (d / 2) ≠ d ∨ bit1 (c / 2) ≠ c ∨ bit1 (b / 2) ≠ b))) ∧
    n = 2016 :=
by
  sorry

end smallest_four_digit_number_divisible_by_8_with_specific_digit_properties_l372_372165


namespace find_a_l372_372058

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372058


namespace solution_l372_372946

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → n % m ≠ 0

def problem_statement : Prop :=
  ∃ p q : Nat, is_prime(p) ∧ is_prime(q) ∧ p + q = 7 ∧ (p^q = 32 ∨ p^q = 25)

theorem solution : problem_statement :=
  sorry

end solution_l372_372946


namespace ellipse_eccentricity_l372_372044

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372044


namespace soup_options_l372_372106

-- Define the given conditions
variables (lettuce_types tomato_types olive_types total_options : ℕ)
variable (S : ℕ)

-- State the conditions
theorem soup_options :
  lettuce_types = 2 →
  tomato_types = 3 →
  olive_types = 4 →
  total_options = 48 →
  (lettuce_types * tomato_types * olive_types * S = total_options) →
  S = 2 :=
by
  sorry

end soup_options_l372_372106


namespace first_term_formula_sum_first_n_terms_b_G_n_value_l372_372071

variable (a b : ℕ → ℕ)

def arithmetic_sequence (a_n X_2 S_5 : ℕ) : Prop := 
  X_2 = 5 ∧ S_5 = 35 ∧ ∀ n, a_n = 2 * n + 1

def geometric_properties (a_n b_n : ℕ) : Prop :=
  ∀ n, a_n = log 2 b_n ∧ b_n = 2 ^ (2 * n + 1)

noncomputable def sum_b (n : ℕ) : ℕ :=
  8 * (4^n - 1) / 3

def G_n (a_n b_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i, a_n i * b_n i)

theorem first_term_formula (a_n : ℕ → ℕ) (X_2 S_5 : ℕ) :
  arithmetic_sequence a_n 5 35 → a_n = λ n, 2 * n + 1 :=
sorry

theorem sum_first_n_terms_b (a_n b_n : ℕ → ℕ) (n : ℕ) :
  geometric_properties a_n b_n → T_n = 8 * (4^n - 1) / 3 :=
sorry

theorem G_n_value (a_n b_n : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a_n 5 35 →
  geometric_properties a_n b_n →
  G_n a_n b_n n = ((48 * n + 8) * 4^n - 8) / 9 :=
sorry

end first_term_formula_sum_first_n_terms_b_G_n_value_l372_372071


namespace find_f_x_l372_372331

def f (x : ℝ) : ℝ := sorry

theorem find_f_x (x : ℝ) (h : 2 * f x - f (-x) = 3 * x) : f x = x := 
by sorry

end find_f_x_l372_372331


namespace max_value_of_expression_l372_372450

theorem max_value_of_expression (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  x^2 * y + x^2 * z + y * z^2 ≤ 3 := sorry

end max_value_of_expression_l372_372450


namespace monthly_rent_calculation_l372_372957

-- Definitions based on the problem conditions
def investment_amount : ℝ := 20000
def desired_annual_return_rate : ℝ := 0.06
def annual_property_taxes : ℝ := 650
def maintenance_percentage : ℝ := 0.15

-- Theorem stating the mathematically equivalent problem
theorem monthly_rent_calculation : 
  let required_annual_return := desired_annual_return_rate * investment_amount
  let total_annual_earnings := required_annual_return + annual_property_taxes
  let monthly_earnings_target := total_annual_earnings / 12
  let monthly_rent := monthly_earnings_target / (1 - maintenance_percentage)
  monthly_rent = 181.38 :=
by
  sorry

end monthly_rent_calculation_l372_372957


namespace quadratic_always_positive_if_and_only_if_l372_372356

theorem quadratic_always_positive_if_and_only_if :
  (∀ x : ℝ, x^2 + m * x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end quadratic_always_positive_if_and_only_if_l372_372356


namespace arithmetic_mean_of_multiples_of_5_l372_372159

-- Define the sequence and its properties
def a₁ : ℕ := 10
def d : ℕ := 5
def aₙ : ℕ := 95

-- Find number of terms in the sequence
def n: ℕ := (aₙ - a₁) / d + 1

-- Define the sum of the sequence
def S := n * (a₁ + aₙ) / 2

-- Define the arithmetic mean
def arithmetic_mean := S / n

-- Prove the arithmetic mean
theorem arithmetic_mean_of_multiples_of_5 : arithmetic_mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_multiples_of_5_l372_372159


namespace domain_of_f_is_interval_neg1_1_f_is_odd_function_solution_set_of_f_gt_0_l372_372716

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (1 + x) - log a (1 - x)

theorem domain_of_f_is_interval_neg1_1
  (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  ∀ x : ℝ, f a x = f a x → x ∈ Ioo (-1:ℝ) 1 := sorry

theorem f_is_odd_function
  (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  ∀ x : ℝ, f a (-x) = - f a x := sorry

theorem solution_set_of_f_gt_0 
  (a : ℝ) (h_pos : 0 < a) (h_lt_one : a < 1) :
  ∀ x : ℝ, f a x > 0 ↔ x ∈ Ioo (-1:ℝ) 0 := sorry

end domain_of_f_is_interval_neg1_1_f_is_odd_function_solution_set_of_f_gt_0_l372_372716


namespace tickets_won_in_skee_ball_l372_372807

-- Define the conditions as Lean definitions
def tickets_from_whack_a_mole : ℕ := 8
def ticket_cost_per_candy : ℕ := 5
def candies_bought : ℕ := 3

-- We now state the conjecture (mathematical proof problem) 
-- Prove that the number of tickets won in skee ball is 7.
theorem tickets_won_in_skee_ball :
  (candies_bought * ticket_cost_per_candy) - tickets_from_whack_a_mole = 7 :=
by
  sorry

end tickets_won_in_skee_ball_l372_372807


namespace tangential_difference_l372_372436

noncomputable def tan_alpha_minus_beta (α β : ℝ) : ℝ :=
  Real.tan (α - β)

theorem tangential_difference 
  {α β : ℝ}
  (h : 3 / (2 + Real.sin (2 * α)) + 2021 / (2 + Real.sin β) = 2024) : 
  tan_alpha_minus_beta α β = 1 := 
sorry

end tangential_difference_l372_372436


namespace triangles_equilateral_l372_372429

def centroid (a b c : ℂ) : Prop := a + b + c = 0

def rotation (angle : ℂ) (z : ℂ) : ℂ := angle * z

def equilateral (a b c : ℂ) : Prop :=
  a^2 + b^2 + c^2 = a * b + b * c + c * a

theorem triangles_equilateral (a b c : ℂ) (h : centroid a b c) :
  let i := complex.exp (2 * real.pi * complex.I / 3)
  let rneg := complex.exp (-2 * real.pi * complex.I / 3)
  let b1 := rotation rneg b
  let c1 := rotation rneg c
  let b2 := rotation i b
  let c2 := rotation i c
  equilateral a b1 c2 ∧ equilateral a b2 c1 :=
by
  sorry

end triangles_equilateral_l372_372429


namespace majority_not_possible_l372_372978

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_nonneg_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem majority_not_possible : 
  (finset.range 1_000_001).filter (λ (n : ℕ), (∃ x y : ℕ, x^2 + y^3 = n)).card 
  < 500_000 := 
sorry

end majority_not_possible_l372_372978


namespace distances_correct_l372_372289

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Defining points A, B, C, D, O, and E
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, 4)
def D : ℝ × ℝ := (5, -2)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (1, -4)

-- The main theorem, asserting the correctness of distances
theorem distances_correct :
  distance (prod.fst A) (prod.snd A) (prod.fst B) (prod.snd B) = 5 ∧
  distance (prod.fst C) (prod.snd C) (prod.fst D) (prod.snd D) = 10 ∧
  distance (prod.fst O) (prod.snd O) (prod.fst E) (prod.snd E) = real.sqrt 17 := by
  sorry

end distances_correct_l372_372289


namespace triangle_area_ratio_l372_372814

theorem triangle_area_ratio (x : ℝ) :
  let ABC_area := (sqrt 3 / 4) * x^2 in
  let A'B'C'_area := (sqrt 3 / 4) * (3 * x)^2 in
  A'B'C'_area / ABC_area = 9 :=
by {
  let ABC_area := (sqrt 3 / 4) * x^2,
  let A'B'C'_area := (sqrt 3 / 4) * (3 * x)^2,
  have h1 : A'B'C'_area = (sqrt 3 / 4) * 9 * x^2 := by sorry,
  have h2 : ABC_area = (sqrt 3 / 4) * x^2 := by sorry,
  calc
    A'B'C'_area / ABC_area
        = ((sqrt 3 / 4) * 9 * x^2) / ((sqrt 3 / 4) * x^2) : by rw [h1, h2]
    ... = 9 : by sorry,
}

end triangle_area_ratio_l372_372814


namespace find_x_squared_l372_372945

theorem find_x_squared (x : ℝ) (h_pos : x > 0) (h_eq : (∛(1 - x^3) + ∛(1 + x^3) = 1)) :
  x^2 = ∛(28) / 3 :=
by
  sorry

end find_x_squared_l372_372945


namespace common_element_in_sets_l372_372846

theorem common_element_in_sets (n : ℕ) (A B : Finset ℕ) 
  (hA : A.card = n) (hAsum : A.sum id = n^2) 
  (hB : B.card = n) (hBsum : B.sum id = n^2) : 
  (A ∩ B).nonempty :=
sorry

end common_element_in_sets_l372_372846


namespace smallest_number_of_eggs_l372_372180

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l372_372180


namespace simplify_expr_l372_372093

-- Define the expression we need to simplify
def expr : ℚ := 1 + (1 / (1 + real.sqrt 5)) + (1 / (1 - real.sqrt 5))

-- State the theorem to demonstrate the simplification
theorem simplify_expr : expr = 1 / 2 := by
  sorry

end simplify_expr_l372_372093


namespace proposition_p_proposition_q_l372_372084

theorem proposition_p (a : ℝ) (ha : a ∈ set.Ioo 0 1 ∪ set.Ioi 1) : Real.log (2 - 1) / Real.log a = 0 := 
by {
  have h : 2 - 1 = 1 := rfl,
  rw [h, Real.log_one, zero_div],
  exact rfl,
}

theorem proposition_q : ¬ ∃ x : ℕ, x^3 < x^2 :=
by {
  intros ⟨x, h⟩,
  rcases x with _ | _ | x;
  { dsimp only at *,
    norm_num at h,
    apply h,
    all_goals {linarith}}
  }

end proposition_p_proposition_q_l372_372084


namespace gimbap_costs_l372_372240

theorem gimbap_costs : (∑ t in ['basic, 'tuna, 'red_pepper, 'beef, 'rice], 
  if t = 'basic then 2000 
  else if t = 'tuna then 3500 
  else if t = 'red_pepper then 3000 
  else if t = 'beef then 4000 
  else if t = 'rice then 3500 else 0) ≥ 3 :=
by sorry

end gimbap_costs_l372_372240


namespace distance_to_campground_l372_372856

theorem distance_to_campground (speed time : ℕ) (h_speed : speed = 60) (h_time : time = 5) :
  speed * time = 300 := 
by
  rw [h_speed, h_time]
  exact rfl

end distance_to_campground_l372_372856


namespace solve_system_l372_372288

variable (y : ℝ) (x1 x2 x3 x4 x5 : ℝ)

def system_of_equations :=
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x3

theorem solve_system :
  (y = 2 → x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∧
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) →
   x1 + x2 + x3 + x4 + x5 = 0 ∧ ∀ (x1 x5 : ℝ), system_of_equations y x1 x2 x3 x4 x5) :=
sorry

end solve_system_l372_372288


namespace problem_l372_372719

theorem problem (a b : ℝ) :
  (∀ x : ℝ, 3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b → -1 ≤ x ∧ x ≤ 2) →
  a + b = 13 := by
  sorry

end problem_l372_372719


namespace sum_of_reciprocals_l372_372262

noncomputable def y : ℕ → ℝ
| 1     => 135
| (k+1) => 2 * (y k)^2 + y k

theorem sum_of_reciprocals (S : ℝ) :
  S = ∑' (k : ℕ), (1 / (y (k + 1) + 1)) → S = 1 / 135 :=
by
  sorry

end sum_of_reciprocals_l372_372262


namespace describe_f_plus_g_l372_372215

open Function

variable (a b c : ℝ)

def parabola (x : ℝ) : ℝ := a * x^2 + b * x + c

def reflected_parabola (x : ℝ) : ℝ := -a * x^2 - b * x - c

def f (x : ℝ) : ℝ := parabola (x - 3)

def g (x : ℝ) : ℝ := reflected_parabola (x + 3)

theorem describe_f_plus_g :
  (∀ a b c : ℝ, a ≠ 0) →
  (∀ x : ℝ, (f + g) x = -6 * (a * x + b)) :=
by
  intros a b c ha
  funext x
  simp [f, g, parabola, reflected_parabola]
  sorry

end describe_f_plus_g_l372_372215


namespace eight_letter_good_words_count_l372_372995

def is_good_word (word : List Char) : Prop :=
  (∀ i < word.length - 1, 
    (word[i] = 'A' → word[i + 1] ≠ 'B' ∧ word[i + 1] ≠ 'D') ∧
    (word[i] = 'B' → word[i + 1] ≠ 'C') ∧
    (word[i] = 'C' → word[i + 1] ≠ 'A') ∧
    (word[i] = 'D' → word[i + 1] ≠ 'A')) ∧
  (∀ i < word.length, word[i] = 'A' ∨ word[i] = 'B' ∨ word[i] = 'C' ∨ word[i] = 'D')

theorem eight_letter_good_words_count : (Finset.filter is_good_word (Finset.range (4^8))).card = 512 :=
sorry

end eight_letter_good_words_count_l372_372995


namespace coeff_x4_expansion_l372_372495

theorem coeff_x4_expansion (C : ℕ → ℕ → ℕ) (binom : ℕ → ℕ → ℕ): 
  (∀ (n k : ℕ), C n k = binom n k) →
  (1*(binom 5 4 * (-2)^4) + 1*(binom 5 2 * (-2)^2) = 120) :=
by 
  intros h
  sorry

end coeff_x4_expansion_l372_372495


namespace binary_product_with_seven_zeros_is_odd_l372_372890

theorem binary_product_with_seven_zeros_is_odd (m : ℕ) (m_le_4_digits : ∃ n : ℕ, n < 2^4 ∧ m = n)
  (binary_10001 : ∀ m, [1,0,0,0,1] = [1,0,0,0,1]) :
  ∃ b, (binary_mul [1,0,0,0,1] (nat_to_binary m) = b ∧ count_digit b 0 7 ∧ is_odd b ∧ b%2=1) := 
sorry

end binary_product_with_seven_zeros_is_odd_l372_372890


namespace absolute_value_z_squared_minus_2z_l372_372754

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem
theorem absolute_value_z_squared_minus_2z : complex.abs (z^2 - 2*z) = 2 := by
  sorry

end absolute_value_z_squared_minus_2z_l372_372754


namespace noncongruent_triangles_square_midpoints_l372_372624

def midpoint {α : Type*} [metric_space α] (x y : α) : α := sorry
def centroid {α : Type*} [metric_space α] (x y z : α) : α := sorry

theorem noncongruent_triangles_square_midpoints :
  ∀ (A B C D P Q R S : Point),
  is_square A B C D →
  is_midpoint A B P →
  is_midpoint B C Q →
  is_midpoint C D R →
  is_midpoint D A S →
  (number_of_noncongruent_triangles [A, B, C, D, P, Q, R, S] = 4) :=
by sorry

end noncongruent_triangles_square_midpoints_l372_372624


namespace find_a_l372_372055

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l372_372055


namespace ellipse_eccentricity_l372_372045

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l372_372045


namespace rhombus_area_l372_372947

def d1 : ℝ := 10
def d2 : ℝ := 30

theorem rhombus_area (d1 d2 : ℝ) : (d1 * d2) / 2 = 150 := by
  sorry

end rhombus_area_l372_372947


namespace maximum_diagonal_intersections_l372_372927

theorem maximum_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
  ∃ k, k = (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
by sorry

end maximum_diagonal_intersections_l372_372927


namespace relatively_prime_dates_in_month_l372_372586

open Nat

theorem relatively_prime_dates_in_month :
  ∃ (n : ℕ), n = 30 ∧ (∀ k : ℕ, k ∈ (Icc 1 n) → gcd k n = 1) → card {k ∈ Icc 1 n | gcd k n = 1} = 8 :=
begin
  let n := 30,
  have div_2 : n % 2 = 0 := by norm_num,
  have div_3 : n % 3 = 0 := by norm_num,
  have div_5 : n % 5 = 0 := by norm_num,
  existsi n,
  refine ⟨rfl, _⟩,
  intro H,
  
  sorry -- proof omitted
end

end relatively_prime_dates_in_month_l372_372586


namespace smallest_prime_10_less_than_perfect_square_l372_372584

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_10_less_than_perfect_square :
  ∃ (a : ℕ), is_prime a ∧ (∃ (n : ℕ), a = n^2 - 10) ∧ (∀ (b : ℕ), is_prime b ∧ (∃ (m : ℕ), b = m^2 - 10) → a ≤ b) ∧ a = 71 := 
by
  sorry

end smallest_prime_10_less_than_perfect_square_l372_372584


namespace clara_weight_l372_372128

-- Define the weights of Alice and Clara
variables (a c : ℕ)

-- Define the conditions given in the problem
def condition1 := a + c = 240
def condition2 := c - a = c / 3

-- The theorem to prove Clara's weight given the conditions
theorem clara_weight : condition1 a c → condition2 a c → c = 144 :=
by
  intros h1 h2
  sorry

end clara_weight_l372_372128


namespace average_seeds_per_apple_l372_372483

-- Define the problem conditions and the proof statement

theorem average_seeds_per_apple
  (A : ℕ)
  (total_seeds_requirement : ℕ := 60)
  (pear_seeds_avg : ℕ := 2)
  (grape_seeds_avg : ℕ := 3)
  (num_apples : ℕ := 4)
  (num_pears : ℕ := 3)
  (num_grapes : ℕ := 9)
  (shortfall : ℕ := 3)
  (collected_seeds : ℕ := num_apples * A + num_pears * pear_seeds_avg + num_grapes * grape_seeds_avg)
  (required_seeds : ℕ := total_seeds_requirement - shortfall) :
  collected_seeds = required_seeds → A = 6 := 
by
  sorry

end average_seeds_per_apple_l372_372483


namespace geometric_progression_sixth_term_proof_l372_372497

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l372_372497


namespace distance_of_course_l372_372899

-- Definitions
def teamESpeed : ℕ := 20
def teamASpeed : ℕ := teamESpeed + 5

-- Time taken by Team E
variable (tE : ℕ)

-- Distance calculation
def teamEDistance : ℕ := teamESpeed * tE
def teamADistance : ℕ := teamASpeed * (tE - 3)

-- Proof statement
theorem distance_of_course (tE : ℕ) (h : teamEDistance tE = teamADistance tE) : teamEDistance tE = 300 :=
sorry

end distance_of_course_l372_372899


namespace color_of_2004th_light_l372_372274

def light_color (n : ℕ) : string :=
  let sequence := ["green", "yellow", "yellow", "red", "red", "red"];
  sequence.get (n % 6)

theorem color_of_2004th_light : light_color 2004 = "red" :=
by
  sorry

end color_of_2004th_light_l372_372274


namespace clearance_sale_gain_percent_l372_372477

theorem clearance_sale_gain_percent
  (SP : ℝ := 30)
  (gain_percent : ℝ := 25)
  (discount_percent : ℝ := 10)
  (CP : ℝ := SP/(1 + gain_percent/100)) :
  let Discount := discount_percent / 100 * SP
  let SP_sale := SP - Discount
  let Gain_during_sale := SP_sale - CP
  let Gain_percent_during_sale := (Gain_during_sale / CP) * 100
  Gain_percent_during_sale = 12.5 := 
by
  sorry

end clearance_sale_gain_percent_l372_372477


namespace coprime_squares_l372_372735

theorem coprime_squares (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ k : ℕ, ab = k^2) : 
  ∃ p q : ℕ, a = p^2 ∧ b = q^2 :=
by
  sorry

end coprime_squares_l372_372735


namespace angle_ADH_eq_angle_BDO_l372_372407

variables {A B C D P M N H O : Type}
variables [point A] [point B] [point C] [point D] [point P]
variables [point M] [point N] [point H] [circle O]

-- Given conditions
variable (circumcircle : circle O)
variable (tangent_line : line)
variable (triangle_ABC : triangle A B C)
variable (D_on_bc_tangent : tangent_circumcircle_cpoint circumcircle A tangent_line D)
variable (P_on_OD : collinear O D P)
variable (M_on_perpendicular_PM_AB : perpendicular_line_line (line_through P M) (line_through A B))
variable (N_on_perpendicular_PN_AC : perpendicular_line_line (line_through P N) (line_through A C))
variable (H_orthocenter_AMN : orthocenter H (triangle A M N))

-- To prove
theorem angle_ADH_eq_angle_BDO : angle A D H = angle B D O :=
sorry

end angle_ADH_eq_angle_BDO_l372_372407


namespace find_speed_taxi_l372_372589

-- Definitions and conditions
def speed_taxi : ℝ := 45
def speed_bus (v : ℝ) : ℝ := v - 30
def distance_taxi (v : ℝ) : ℝ := v * 2
def distance_bus (v : ℝ) : ℝ := (v - 30) * 6

-- Theorem statement
theorem find_speed_taxi (v : ℝ) (h1 : distance_taxi v = distance_bus v) : speed_taxi = v :=
by
  have : 2 * v = 6 * (v - 30) := by assumption
  sorry

end find_speed_taxi_l372_372589


namespace average_salary_all_workers_l372_372109

theorem average_salary_all_workers (T R : ℕ) (A_tech A_rest : ℝ) (total_workers : ℕ)
  (h1 : T = 10) (h2 : A_tech = 1200) (h3 : R = 11) (h4 : A_rest = 820) (h5 : total_workers = 21)
  :
  let total_salary := (T * A_tech) + (R * A_rest)
  let avg_salary := total_salary / total_workers
  in avg_salary = 1002 :=
by
  sorry

end average_salary_all_workers_l372_372109


namespace combined_weight_of_two_new_men_l372_372493

theorem combined_weight_of_two_new_men
    (avg_increase : 2.5)
    (replaced_man_weight : 68)
    (new_group_size : 11)
    (total_weight_increase : 2.5 * 11 = 27.5)
    : W = 95.5 :=
by 
  sorry

end combined_weight_of_two_new_men_l372_372493


namespace number_of_incorrect_steps_l372_372268

theorem number_of_incorrect_steps 
  (a b c d : ℝ) 
  (h₁ : a > b) 
  (h₂ : c > d) 
  (step1_correct : ¬ (a > b ∧ c > d → ac > bc ∧ bc > bd)) 
  (step2_correct : a * c > b * c → b * c > b * d → a * c > b * d)
  (step3_correct : ¬ (a * c > b * d → a / d > b / c)) 
: 2 = 2 := 
sorry

end number_of_incorrect_steps_l372_372268


namespace total_days_spent_on_island_l372_372803

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l372_372803


namespace hexagon_circumscribable_l372_372466

-- Let the hexagon be represented as a list of points in the plane.
structure Hexagon (α : Type*) [AddCommGroup α] [Module ℝ α] :=
(points : fin 6 → α)

-- Define the condition for opposite sides being parallel.
def opposite_sides_parallel {α : Type*} [InnerProductSpace ℝ α] (h : Hexagon α) : Prop :=
(∀ i : fin 3, inner (h.points i -ᵥ h.points (i + 1)) (h.points (i + 3) -ᵥ h.points (i + 4)) = 0)

-- Define the condition for diagonals connecting opposite vertices being equal.
def diagonals_equal {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (h : Hexagon α) : Prop :=
(∀ i : fin 3, ∥h.points i -ᵥ h.points (i + 3)∥ = ∥h.points (i + 1) -ᵥ h.points (i + 4)∥)

-- The theorem stating that under the given conditions, the hexagon can be circumscribed by a circle.
theorem hexagon_circumscribable {α : Type*} [InnerProductSpace ℝ α] (h : Hexagon α)
  (hp : opposite_sides_parallel h) (hd : diagonals_equal h) : 
  ∃ O : α, ∀ i : fin 6, ∥h.points i -ᵥ O∥ = ∥h.points 0 -ᵥ O∥ :=
sorry

end hexagon_circumscribable_l372_372466


namespace right_triangle_ellipse_l372_372991

theorem right_triangle_ellipse
  (a b c : ℝ)
  (Q : ℝ × ℝ) 
  (PR_parallel_x : ((-a, 0) : ℝ×ℝ) × ((a, 0) : ℝ×ℝ)
  (f1 f2 : ℝ × ℝ)
  (ellipse_eq : (Q.1)^2 / a^2 + (Q.2)^2 / b^2 = 1)
  (f1_on_qr : f1 on segment QR)
  (f2_on_pq : f2 on segment PQ)
  (dist_f1f2 : dist f1 f2 = 2)
  (a_sq_b_sqrt3 : a^2 - b^2 = c^2)
  (c_1 : c = 1)
  (q : Q = (0, b))
  (p : P = (-a, 0) ) 
  (r : R = (a, 0))
  : abs_dist PQ dist(f1, f2) = (sqrt(7)) / 2 := 
  sorry

end right_triangle_ellipse_l372_372991


namespace initial_apples_l372_372607

-- Define the initial conditions
def r : Nat := 14
def s : Nat := 2 * r
def remaining : Nat := 32
def total_removed : Nat := r + s

-- The proof problem: Prove that the initial number of apples is 74
theorem initial_apples : (total_removed + remaining = 74) :=
by
  sorry

end initial_apples_l372_372607


namespace cone_hill_depth_l372_372933

theorem cone_hill_depth (h : ℝ) (Vab : ℝ) (h_total : h = 5000)
  (Vab_eq : Vab = h^3 * π / 3)
  (top_volume_frac : 1 / 5 * Vab = Vab - ((4 / 5) * Vab)) :
  (h * (1 - real.cbrt (4 / 5)) = 355) :=
by 
-- declare constants
let k := real.cbrt(4 / 5)
-- declare equation
have h_submerged_eq : (1 - k) * h = 355,
-- substitution by given conditions
rw [h_total, k],
sorry

end cone_hill_depth_l372_372933


namespace energy_difference_l372_372894

theorem energy_difference : 
  ∃ (q : ℝ), 
  let A := (0, 0),
      B := (1, 0),
      C := (0, 1),
      M := ((B.1 + C.1)/2, (B.2 + C.2)/2),
      initial_energy := 18,
      energy_AB := (initial_energy / 3),
      energy_BC := (initial_energy / 3),
      energy_CA := (initial_energy / 3),
      distance_BM := sqrt 2 / 2 in
  q = initial_energy - (energy_AB + 2 * energy_AB * sqrt 2 / distance_BM) :=
sorry

end energy_difference_l372_372894


namespace product_of_t_values_l372_372663

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372663


namespace tangent_line_at_P_l372_372873

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 1

def derivative_f (x : ℝ) : ℝ := (deriv f) x

theorem tangent_line_at_P : 
    tangent_at f (derivative_f) (-1, 3) = λ x, -4 * x - 1 := by
    sorry

end tangent_line_at_P_l372_372873


namespace ellipse_a_value_l372_372030

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372030


namespace pairwise_sum_bounds_l372_372558

theorem pairwise_sum_bounds (n : ℕ) (a : Fin n → ℝ) (d s : ℝ) 
  (h1 : ∀ i j : Fin n, i < j → |a j - a i| ≤ d) 
  (h2 : ∑ i in (Finset.range n).filter (finset.completed (a j - a i)) (if filter i < j on absolute |a j - a i) = s) 
  (h3 : a ≠ ∅ ∧ sorted a a_0 a_(n-1) d = a_(n-1) - a_0) : 
  (n - 1) * d ≤ s ∧ s ≤ (n^2 / 4) * d :=
by
  sorry

end pairwise_sum_bounds_l372_372558


namespace intersection_A_B_union_B_Ac_range_a_l372_372360

open Set

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def Ac : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def Bc : Set ℝ := {x | x < -2 ∨ x > 5}

-- Questions rewritten as Lean statements

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_Ac :
  B ∪ Ac = {x | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_a (a : ℝ) :
  {x | a ≤ x ∧ x ≤ a + 2} ⊆ Bc → a ∈ Iio (-4) ∪ Ioi 5 := sorry

end intersection_A_B_union_B_Ac_range_a_l372_372360


namespace ellipse_eccentricity_a_l372_372007

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l372_372007


namespace head_start_correct_l372_372204

noncomputable def head_start (v : ℝ) := 
  let s := 84 - 21 in
  s

theorem head_start_correct (v : ℝ) (hv : v > 0) : head_start v = 63 := 
by
  calc
    head_start v = 84 - 21 := by rfl
    ... = 63 := by linarith

end head_start_correct_l372_372204


namespace ellipse_a_value_l372_372029

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l372_372029


namespace river_flow_volume_l372_372948

/-- Given a river depth of 7 meters, width of 75 meters, 
and flow rate of 4 kilometers per hour,
the volume of water running into the sea per minute 
is 35,001.75 cubic meters. -/
theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (rate_kmph : ℝ)
  (depth_val : depth = 7)
  (width_val : width = 75)
  (rate_val : rate_kmph = 4) :
  ( width * depth * (rate_kmph * 1000 / 60) ) = 35001.75 :=
by
  rw [depth_val, width_val, rate_val]
  sorry

end river_flow_volume_l372_372948


namespace books_sold_at_overall_loss_l372_372367

-- Defining the conditions and values
def total_cost : ℝ := 540
def C1 : ℝ := 315
def loss_percentage_C1 : ℝ := 0.15
def gain_percentage_C2 : ℝ := 0.19
def C2 : ℝ := total_cost - C1
def loss_C1 := (loss_percentage_C1 * C1)
def SP1 := C1 - loss_C1
def gain_C2 := (gain_percentage_C2 * C2)
def SP2 := C2 + gain_C2
def total_selling_price := SP1 + SP2
def overall_loss := total_cost - total_selling_price

-- Formulating the theorem based on the conditions and required proof
theorem books_sold_at_overall_loss : overall_loss = 4.50 := 
by 
  sorry

end books_sold_at_overall_loss_l372_372367


namespace no_division_into_9_equal_parts_l372_372893

-- Define the problem
theorem no_division_into_9_equal_parts :
  ∀ (S : set (ℝ × ℝ)) (P Q : ℝ × ℝ),
  (square S) ∧ (in_square S P) ∧ (in_square S Q) →
  ¬(division_into_9_equal_parts S P Q) :=
by
  sorry

-- Definitions as per the conditions in the problem
def square (S : set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ), is_square S a b c d

def in_square (S : set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  P ∈ S

def division_into_9_equal_parts (S : set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Prop :=
  ∃ (regions : list (set (ℝ × ℝ))), 
  (∀ (r : set (ℝ × ℝ)), r ∈ regions → area r = area S / 9) ∧
  (⋃₀ regions = S) ∧
  (∀ (r1 r2 : set (ℝ × ℝ)), r1 ≠ r2 → r1 ∩ r2 = ∅) ∧
  connects_via_segments_to_vertices S P Q regions

def is_square (S : set (ℝ × ℝ)) (a b c d : ℝ × ℝ) : Prop :=
  -- Some specific definition to ensure (a, b, c, d) form a square

def connects_via_segments_to_vertices (S : set (ℝ × ℝ)) (P Q : ℝ × ℝ) (regions : list (set (ℝ × ℝ))) : Prop :=
  -- Specific definition to state how P and Q connect to vertices

end no_division_into_9_equal_parts_l372_372893


namespace positive_difference_prime_factors_165033_l372_372537

def largest_prime_factors (n : ℕ) : ℕ × ℕ := sorry

theorem positive_difference_prime_factors_165033 :
  let n := 165033
  let (p1, p2) := largest_prime_factors n
  in p1 - p2 = 140 :=
by
  sorry

end positive_difference_prime_factors_165033_l372_372537


namespace coefficient_of_x3_in_binomial_expansion_l372_372265

theorem coefficient_of_x3_in_binomial_expansion : 
  let expr := (x - (2/x))^5,
  ∃ C : ℤ, C = -10 ∧ (C * x^3) ∈ expr.expand(x) := 
sorry

end coefficient_of_x3_in_binomial_expansion_l372_372265


namespace mul_inv_mod_391_l372_372258

theorem mul_inv_mod_391 (a : ℤ) (ha : 143 * a % 391 = 1) : a = 28 := by
  sorry

end mul_inv_mod_391_l372_372258


namespace variance_comparison_l372_372622

noncomputable def average (s : Finset ℝ) : ℝ :=
  s.sum / s.card

noncomputable def variance (s : Finset ℝ) (avg : ℝ) : ℝ :=
  (s.sum (λ x, (x - avg) ^ 2)) / s.card

theorem variance_comparison (s : Finset ℝ) 
  (h_card : s.card = 5) 
  (t : Finset ℝ) 
  (h_t_subset_s : t ⊂ s) 
  (h_t_card : t.card = 4) :
  let avg_s := average s in
  let var_s := variance s avg_s in
  let avg_t := average t in
  let var_t := variance t avg_t in
  avg_s = avg_t → var_s < var_t :=
by
  sorry

end variance_comparison_l372_372622


namespace xy_eq_xh_l372_372809

-- Definitions based on the conditions
structure Square (ABCD : ℝ × ℝ → Prop) :=
  (is_square : ∀ (A B C D : ℝ × ℝ), ABCD A ∧ ABCD B ∧ ABCD C ∧ ABCD D 
                  → (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A)
                    ∧ ∀ (A' B' : ℝ × ℝ), A ≠ B → ABCD A' → ABCD B' → (dist A' B') > 0)

structure PointOnLineSegment (A B H : ℝ × ℝ) :=
  (condition : dist A B * dist B H = dist A H ^ 2)

def is_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  (M.fst = (P1.fst + P2.fst) / 2) ∧ (M.snd = (P1.snd + P2.snd) / 2)

def is_perpendicular (P1 P2 P3 : ℝ × ℝ) : Prop :=
  (P2.snd - P1.snd) * (P3.snd - P2.snd) = - (P2.fst - P1.fst) * (P3.fst - P2.fst)

noncomputable def point_on_segment (E B : ℝ × ℝ) : Type := { Y : ℝ × ℝ // true }  -- Y is just an unspecified point on EB

theorem xy_eq_xh {A B C D H E X Y : ℝ × ℝ}
  (sq : Square (λ P, P = A ∨ P = B ∨ P = C ∨ P = D))
  (hH : PointOnLineSegment A B H)
  (hE : is_midpoint A D E)
  (hX : is_midpoint A H X)
  (hY_exists : ∃ Y, is_perpendicular X Y E ∧ is_perpendicular Y B E) :
  dist X Y = dist X H :=
sorry

end xy_eq_xh_l372_372809


namespace total_rowing_proof_l372_372564

def morning_rowing := 13
def afternoon_rowing := 21
def total_rowing := 34

theorem total_rowing_proof :
  morning_rowing + afternoon_rowing = total_rowing :=
by
  sorry

end total_rowing_proof_l372_372564


namespace hare_hit_within_four_volleys_l372_372192

universe u

structure Cube :=
(vertices : Fin 8)
(edges : vertices → vertices → Prop)

structure ShootingSeq :=
(shots : Fin 4 → Fin 3 → Cube)

noncomputable def hunters_strategy (c : Cube) : ShootingSeq :=
{ shots := λ i, match i with
                | 0 => λ j, [2, 5, 7].nth j ⟨sorry, sorry⟩ -- (C, F, H)
                | 1 => λ j, [1, 3, 4].nth j ⟨sorry, sorry⟩ -- (B, D, E)
                | 2 => λ j, [3, 4, 6].nth j ⟨sorry, sorry⟩ -- (D, E, G)
                | 3 => λ j, [0, 2, 5].nth j ⟨sorry, sorry⟩ -- (A, C, F)
                end
}

theorem hare_hit_within_four_volleys (c : Cube) (s : ShootingSeq) :
  (∀ v : c.vertices, (∃ (i : Fin 4), ∃ (j : Fin 3), s.shots i j = v) →
  (∃ (i : Fin 4), ∃ (j : Fin 3), s.shots i j = v)) :=
by
  intros v h
  existsi (!0)
  existsi (!0)
  sorry

end hare_hit_within_four_volleys_l372_372192


namespace coefficient_x2_neg20_l372_372868

def poly := (x - 1) - (x - 1)^2 + (x - 1)^3 - (x - 1)^4 + (x - 1)^5

theorem coefficient_x2_neg20 :
  (coeff (expand poly) 2 = -20) :=
sorry

end coefficient_x2_neg20_l372_372868


namespace center_of_circle_l372_372870

theorem center_of_circle (x y : ℝ) :
  (x - 1)^2 + (y - 2)^2 = 1 → (x, y) = (1, 2) :=
by
  intro h
  have h₁ : x = 1 :=
    sorry -- This would be provided in the actual proof
  have h₂ : y = 2 :=
    sorry -- This would be provided in the actual proof
  exact eq_of_heq (HEq.intro h₁ h₂)

end center_of_circle_l372_372870


namespace solve_for_a_l372_372001

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l372_372001


namespace solve_for_n_l372_372999

theorem solve_for_n :
  ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 :=
by
  sorry

end solve_for_n_l372_372999


namespace a_5_value_l372_372791

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1/2
  else if n > 0 then 
    let a : ℕ → ℝ := λ n, if n = 1 then 1/2 else 1 - (1/(sequence (n - 1))) in
    a n
  else 0

theorem a_5_value :
  sequence 4 = -1 :=
sorry

end a_5_value_l372_372791


namespace when_to_sell_goods_l372_372609

variable (a : ℝ) (currentMonthProfit nextMonthProfitWithStorage : ℝ) 
          (interestRate storageFee thisMonthProfit nextMonthProfit : ℝ)
          (hm1 : interestRate = 0.005)
          (hm2 : storageFee = 5)
          (hm3 : thisMonthProfit = 100)
          (hm4 : nextMonthProfit = 120)
          (hm5 : currentMonthProfit = thisMonthProfit + (a + thisMonthProfit) * interestRate)
          (hm6 : nextMonthProfitWithStorage = nextMonthProfit - storageFee)

theorem when_to_sell_goods :
  (a > 2900 → currentMonthProfit > nextMonthProfitWithStorage) ∧
  (a = 2900 → currentMonthProfit = nextMonthProfitWithStorage) ∧
  (a < 2900 → currentMonthProfit < nextMonthProfitWithStorage) := by
  sorry

end when_to_sell_goods_l372_372609


namespace geometric_progression_properties_l372_372501

-- Define the first term and the fifth term given
def b₁ := Real.sqrt 3
def b₅ := Real.sqrt 243

-- Define the nth term formula for geometric progression
def geometric_term (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

-- State both the common ratio and the sixth term
theorem geometric_progression_properties :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ 
           geometric_term b₁ q 5 = b₅ ∧ 
           geometric_term b₁ q 6 = 27 ∨ geometric_term b₁ q 6 = -27 :=
by
  sorry

end geometric_progression_properties_l372_372501


namespace product_of_solutions_product_of_all_t_l372_372675

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l372_372675


namespace range_of_x_l372_372328

open Set

noncomputable def f : ℝ → ℝ := sorry

axiom mon_increasing_on_pos : ∀ {x y : ℝ}, (0 < x ∧ 0 < y) → (x < y → f(x) < f(y))

theorem range_of_x (h : ∀ x ∈ (0:ℝ, ∞), f(x) > f(2 - x)) : Ioc 1 2 = { x : ℝ | ∃ a b : ℝ, x = a ∧ 1 < a ∧ a < b ∧ a + b = 2 } :=
by
  sorry

end range_of_x_l372_372328


namespace product_of_t_values_l372_372661

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l372_372661


namespace y_greater_than_one_l372_372121

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l372_372121


namespace solve_for_k_l372_372272

theorem solve_for_k (k : ℝ) : (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 :=
  sorry

end solve_for_k_l372_372272


namespace product_of_solutions_of_t_squared_eq_49_l372_372667

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l372_372667


namespace two_distinct_real_roots_l372_372884

-- Define the variables and coefficients for the quadratic equation
def a : ℝ := 1
def b : ℝ := -4
def c : ℝ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- State the theorem that the discriminant is greater than zero
theorem two_distinct_real_roots : discriminant a b c > 0 := by
  calc
    discriminant a b c = b^2 - 4 * a * c : by rfl
    ...               = (-4)^2 - 4 * 1 * 2 : by simp [a, b, c]
    ...               = 16 - 8 : by simp
    ...               = 8 : by simp
    ...               > 0 : by norm_num

end two_distinct_real_roots_l372_372884


namespace problem_l372_372309

variables (x y z : ℝ)

theorem problem :
  x - y - z = 3 ∧ yz - xy - xz = 3 → x^2 + y^2 + z^2 = 3 :=
by
  sorry

end problem_l372_372309


namespace cube_division_l372_372574

/-- A cube of edge length 4 cm is cut into smaller cubes of integer edge lengths, 
    and not all the smaller cubes are the same size. -/
theorem cube_division (N : ℕ) :
  (∀ (a : ℕ), a ∣ 4) ∧ 
  N = ∑ i in {1, 2, 4}, multiplicity i N → 
  N = 57 :=
begin
  sorry
end

end cube_division_l372_372574


namespace square_pyramid_sum_l372_372545

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end square_pyramid_sum_l372_372545


namespace pirate_loot_l372_372533

theorem pirate_loot (a b c d e : ℕ) (h1 : a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1 ∨ e = 1)
  (h2 : a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 ∨ e = 2)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h4 : a + b = 2 * (c + d) ∨ b + c = 2 * (a + e)) :
  (a, b, c, d, e) = (1, 1, 1, 1, 2) ∨ 
  (a, b, c, d, e) = (1, 1, 2, 2, 2) ∨
  (a, b, c, d, e) = (1, 2, 3, 3, 3) ∨
  (a, b, c, d, e) = (1, 2, 2, 2, 3) :=
sorry

end pirate_loot_l372_372533


namespace range_of_a_circle_C_intersects_circle_D_l372_372728

/-- Definitions of circles C and D --/
def circle_C_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def circle_D_eq (x y m : ℝ) := x^2 + y^2 - 2 * m * x = 0

/-- Condition for the line intersecting Circle C --/
def line_intersects_circle_C (a : ℝ) := (∃ x y : ℝ, circle_C_eq x y ∧ (x + y = a))

/-- Proof of range for a --/
theorem range_of_a (a : ℝ) : line_intersects_circle_C a → (2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2) :=
sorry

/-- Proposition for point A lying on circle C and satisfying the inequality --/
def point_A_on_circle_C_and_inequality (m : ℝ) (x y : ℝ) :=
  circle_C_eq x y ∧ x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

/-- Proof that Circle C intersects Circle D --/
theorem circle_C_intersects_circle_D (m : ℝ) (a : ℝ) : 
  (∀ (x y : ℝ), point_A_on_circle_C_and_inequality m x y) →
  (1 ≤ m ∧
   ∃ (x y : ℝ), (circle_D_eq x y m ∧ (Real.sqrt ((m - 1)^2 + 1) < m + 1 ∧ Real.sqrt ((m - 1)^2 + 1) > m - 1))) :=
sorry

end range_of_a_circle_C_intersects_circle_D_l372_372728


namespace no_adjacent_empty_seats_exactly_three_adjacent_empty_seats_at_most_two_adjacent_empty_seats_l372_372524

-- 1. Number of ways they can sit such that no empty seats are adjacent.
theorem no_adjacent_empty_seats : 
  ∃ (n : ℕ), n = 25200 ∧ 
  (∀ (seats : vector ℕ 10) (people : ℕ), 
  people = 6 ∧ 
  (∀ i j, i ≠ j → seats.nth i ≠ seats.nth j) → 
  -- The exact distribution condition that encodes no adjacent
  (∀ i, (seats.nth i = 0) → (seats.nth (i + 1) ≠ 0))) := 
sorry

-- 2. Number of ways they can sit such that exactly 3 out of the 4 empty seats are adjacent.
theorem exactly_three_adjacent_empty_seats : 
  ∃ (n : ℕ), n = 30240 ∧ 
  (∀ (seats : vector ℕ 10) (people : ℕ), 
  people = 6 ∧ 
  (∀ i j, i ≠ j → seats.nth i ≠ seats.nth j) →
  -- The exact distribution condition that ensures exactly three out of four adjacent
  (∃ (i : ℕ), (seats.nth i = 0) ∧ (seats.nth (i + 1) = 0) ∧ (seats.nth (i + 2) = 0) ∧ (seats.nth (i + 3) = 0) ∧ (∀ j, j ≠ i → (seats.nth j = 0 → seats.nth (j + 1) ≠ 0)))) := 
sorry

-- 3. Number of ways they can sit such that at most 2 out of the 4 empty seats are adjacent.
theorem at_most_two_adjacent_empty_seats : 
  ∃ (n : ℕ), n = 115920 ∧ 
  (∀ (seats : vector ℕ 10) (people : ℕ), 
  people = 6 ∧ 
  (∀ i j, i ≠ j → seats.nth i ≠ seats.nth j) →
  -- The exact distribution condition that ensures at most two out of four adjacent
  (∀ i, (seats.nth i = 0 → seats.nth (i + 1) ≠ 0 → seats.nth (i + 2) ≠ 0))) := 
sorry

end no_adjacent_empty_seats_exactly_three_adjacent_empty_seats_at_most_two_adjacent_empty_seats_l372_372524


namespace problem_solution_l372_372600

def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≤ f y

def is_even_and_monotonically_increasing 
  (f : ℝ → ℝ) (I : set ℝ) : Prop :=
is_even f ∧ is_monotonically_increasing f I

theorem problem_solution :
  is_even_and_monotonically_increasing (λ x, |x| + 1) (set.Ioi 0) :=
sorry

end problem_solution_l372_372600


namespace pictures_total_l372_372088

theorem pictures_total (peter_pics : ℕ) (quincy_extra_pics : ℕ) (randy_pics : ℕ) (quincy_pics : ℕ) (total_pics : ℕ) 
  (h1 : peter_pics = 8)
  (h2 : quincy_extra_pics = 20)
  (h3 : randy_pics = 5)
  (h4 : quincy_pics = peter_pics + quincy_extra_pics)
  (h5 : total_pics = randy_pics + peter_pics + quincy_pics) :
  total_pics = 41 :=
by sorry

end pictures_total_l372_372088


namespace as_eq_bs_of_geometric_properties_l372_372464

theorem as_eq_bs_of_geometric_properties
  (A B C K S : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace S]
  [AffinelyIndependent ℝ ![A, B, C]]
  (angle_ACB_eq_45 : ∠ A C B = π / 4)
  (AK_eq_2KC : dist A K = 2 * dist K C)
  (K_on_AC : ∃ t : ℝ, t ∈ Icc 0 1 ∧ K = (1 - t) • A + t • C) 
  (S_on_BK : ∃ t : ℝ, t ∈ Icc 0 1 ∧ S = (1 - t) • B + t • K)
  (AS_perp_BK : dist (orthm (line_through A S) (line_through B K)) 0 = 0)
  (angle_AKS_eq_60 : ∠ A K S = π / 3) :
  dist A S = dist B S := sorry

end as_eq_bs_of_geometric_properties_l372_372464


namespace basketball_team_win_proof_l372_372568

def basketball_team_win_probability (total_games first_wins remaining_games must_win_percentage remaining_wins : ℕ) :=
  total_games = first_wins + remaining_games ∧
  (must_win_percentage / 100 : ℚ) = 3 / 4 ∧
  (first_wins + remaining_wins) / total_games = must_win_percentage / 100

theorem basketball_team_win_proof :
  basketball_team_win_probability 100 35 55 75 40 :=
by
  unfold basketball_team_win_probability
  split
  rfl
  split
  norm_num
  calc
    (35 + 40 : ℚ) / 100 = 75 / 100 : by norm_num
                      ... = 3 / 4   : by norm_num

end basketball_team_win_proof_l372_372568


namespace polynomial_symmetry_inequality_l372_372707

-- Define the polynomial function f(x)
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c

-- Main theorem statement
theorem polynomial_symmetry_inequality (b c : ℝ)
  (h : ∀ x : ℝ, f(x) b c = f(3 + x) b c = f(3-x) b c) :
  f(4) b c < f(1) b c ∧ f(1) b c < f(-1) b c :=
  sorry

end polynomial_symmetry_inequality_l372_372707


namespace trapezoid_area_l372_372235

def isosceles_trapezoid_area (a b c d diag₁ diag₂ : ℝ) (ha : a = 40) (hb : b = 40) (hc : c = 60) (hd : d = 15) (hdiag₁ : diag₁ = 50) (hdiag₂ : diag₂ = 50) : ℝ :=
  (1 / 2) * (c + d) * sqrt ((diag₁ ^ 2 - ((d - c) / 2) ^ 2))

theorem trapezoid_area (a b c d diag₁ diag₂ : ℝ)
  (ha : a = 40) (hb : b = 40) (hc : c = 60) (hd : d = 15) (hdiag₁ : diag₁ = 50) (hdiag₂ : diag₂ = 50) :
  isosceles_trapezoid_area a b c d diag₁ diag₂ ha hb hc hd hdiag₁ hdiag₂ = 1242.425 :=
sorry

end trapezoid_area_l372_372235
